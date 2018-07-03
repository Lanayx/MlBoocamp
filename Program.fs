﻿// Learn more about F# at http://fsharp.org

open System
open System.IO
open System.Collections.Generic
open Newtonsoft.Json
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers
open Microsoft.ML.Runtime.Api;
open System.Globalization

let getMetric metric =
    let obj = JsonConvert.DeserializeObject<Dictionary<string, int>>(metric)
    obj

let appendDictionary (main: Dictionary<string, (int*int)>) (adding: Dictionary<string, int>) prefix =
    for elem in adding do
        let key = prefix + elem.Key
        if (main.ContainsKey key)
        then
            let (sum, num) = main.[key]
            main.[key] <- (sum + elem.Value, num + 1)
        else
            main.[key] <- (elem.Value, 1)

let extractMetrics (metric: Dictionary<string, int>) =
    metric.Keys

let addMetricToSet (dict: Dictionary<string, int>) (goodSet: HashSet<string>) prefix metric =
    getMetric metric
    |> extractMetrics
    |> Seq.map (fun el -> prefix + el)
    |> Seq.iter (fun metricName ->
        let mutable count = 0
        if dict.TryGetValue(metricName, &count)
        then
            count <- count + 1
            dict.[metricName] <- count
            if count = 10000
            then goodSet.Add(metricName) |> ignore
        else
            dict.[metricName] <- 1)

let loadDataNew() =

    let dict = Dictionary<string, byte>()
    let metricDictionary = Dictionary<string, int>(StringComparer.InvariantCultureIgnoreCase)
    let goodMetricSet = HashSet<string>(StringComparer.InvariantCultureIgnoreCase)
    goodMetricSet.Add("label") |> ignore

    let addMetric = addMetricToSet metricDictionary goodMetricSet
    let mutable counter = 0

    let linesWithAnswers  = File.ReadLines("../../../data/mlboot_train_answers.tsv")
    for line in linesWithAnswers do
        let values = line.Split('\t')
        if
            values.[0] <> "cuid"
        then
            dict.[values.[0]] <- values.[1] |> byte

    let linesWithData  = File.ReadLines("../../../data/mlboot_data.tsv")
    for line in linesWithData do
        counter <- counter + 1
        let res = line.Split('\t')
        let [|id; category; metric1; metric2; metric3; days |] = res
        if dict.ContainsKey(id)
        then
            addMetric "x" metric1
            addMetric "y" metric2
            addMetric "z" metric3
    [0..5] |> List.iter (fun i -> goodMetricSet.Add("cat"+i.ToString()) |> ignore)
    goodMetricSet.Add("days") |> ignore
    goodMetricSet

let getTrainUsers() =
    let dict = Dictionary<string, int>()
    let linesWithAnswers  = File.ReadLines("../../../data/mlboot_train_answers.tsv")
    for line in linesWithAnswers do
        let values = line.Split('\t')
        if
            values.[0] <> "cuid"
        then
            dict.[values.[0]] <- values.[1] |> int
    dict


let fillDataSet (dict: Dictionary<string, int>) headersPath dataPath srPath f writeHeaders =
    let mutable counter = 0
    let headersString = File.ReadAllText(headersPath)
    let headers = headersString.Split(',')

    let linesWithData  = File.ReadLines(dataPath)
    File.WriteAllText(srPath, String.Empty);
    use fs = File.OpenWrite(srPath)
    use sr = new StreamWriter(fs)
    sr.AutoFlush <- true
    if writeHeaders
    then
        sr.WriteLine(headersString)

    let userValues = Dictionary<string, int>(StringComparer.InvariantCultureIgnoreCase)
    let userValuesCounter = Dictionary<string, (int*int)>()
    headers |> Seq.iter (fun elem -> userValues.Add(elem, 0))

    let mutable currentId = ""
    let mutable touched = false

    for line in linesWithData do
        let res = line.Split('\t')
        let [| id; category; metric1; metric2; metric3; days |] = res
        if dict.ContainsKey(id)
        then
            if currentId <> id && currentId <> ""
            then
                for kv in userValuesCounter do
                    let (sum, count)= kv.Value
                    if userValues.ContainsKey(kv.Key)
                    then
                        userValues.[kv.Key] <- sum / count
                        touched <- true

                if touched
                then
                    f id userValues sr
                    counter <- counter + 1
                    if counter % 100 = 0
                    then printfn "%A done" counter
                    touched <- false
                    userValues.Clear()
                    userValuesCounter.Clear()
                    headers |> Seq.iter (fun elem -> userValues.Add(elem, 0))

            currentId <- id
            let storeValues metric prefix =
                appendDictionary userValuesCounter (metric |> getMetric) prefix

            storeValues metric1 "x"
            storeValues metric2 "y"
            storeValues metric3 "z"
            if userValuesCounter.ContainsKey("days")
            then
                let (sum, count) = userValuesCounter.["days"]
                userValuesCounter.["days"] <- (sum + (days |> int), count + 1)
            else
                userValuesCounter.["days"] <- (days |> int, 1)

            userValues.["cat" + category.ToString()] <- 1
            userValues.["label"] <- (int) dict.[id]

let extractTestUsers() =
    let dict = Dictionary<string, int>()
    let linesWithAnswers  = File.ReadLines("../../../data/mlboot_test.tsv")
    for line in linesWithAnswers do
        let values = line.Split('\t')
        if
            values.[0] <> "cuid"
        then
            dict.[values.[0]] <- 1

    let linesWithData  = File.ReadLines("../../../data/mlboot_data.tsv")
    use fs = File.OpenWrite("../../../result/xusers_data.csv")
    use sr = new StreamWriter(fs)
    for line in linesWithData do
        let res = line.Split('\t')
        let [| id; category; metric1; metric2; metric3; days |] = res
        if dict.ContainsKey(id)
        then
            sr.WriteLine(line)

let getInputs() =
    let dict = Dictionary<string, int>()
    let linesWithAnswers  = File.ReadLines("../../../data/mlboot_test.tsv")
    for line in linesWithAnswers do
        if
            line <> "cuid"
        then
            dict.[line] <- 0
    dict


type Input =
        [<Column("0", "Label")>]
        val mutable Label: single
        [<Column("1", "Features")>]
        [<VectorType(14699)>]
        val mutable Features: single[]

        new() = { Label = 0.0f; Features = [||] }


type Output =

        [<ColumnName("Score")>]
        val mutable Score: single

        new() = { Score = 0.0f; }


let writeDataSetRow id (userValues: Dictionary<string, int>) (sr: StreamWriter) =
    let str = String.Join(',', userValues.Values)
    sr.WriteLine(str)

let writePrediction (model: PredictionModel<Input, Output>) id (userValues: Dictionary<string, int>) (sr: StreamWriter) =
    let label = userValues.["label"] |> single
    userValues.Remove("label") |> ignore
    let input = new Input()
    input.Label <- label
    input.Features <- userValues.Values |> Seq.map (fun x -> (single) x) |> Seq.toArray
    let output = model.Predict(input)
    let score = if output.Score > 0.0f then output.Score else 0.0f
    sr.WriteLine(id + "," + score.ToString(CultureInfo.InvariantCulture))

let extractResults() =
    let list = ResizeArray<string>()
    let dict = Dictionary<string, single>()
    let testUsers  = File.ReadLines("../../../data/mlboot_test1.tsv")
    for line in testUsers do
        list.Add(line)

    let answers  = File.ReadLines("../../../result/tempResults.csv")
    for line in answers do
        let res = line.Split(',')
        let [| id; value |] = res
        dict.Add(id, value |> single )

    File.WriteAllText("../../../result/final_res.csv", String.Empty);
    use fs = File.OpenWrite("../../../result/final_res.csv")
    use sr = new StreamWriter(fs)
    for id in list do
        if (dict.ContainsKey(id))
        then sr.WriteLine(dict.[id].ToString(CultureInfo.InvariantCulture))
        else
            printfn "%A is missing" id
            sr.WriteLine("0")

[<EntryPoint>]
let main argv =

    //let headers = loadDataNew()
    //File.WriteAllText("../../../result/headers10.csv", String.Join(",",headers))

    //fillDataSet
    //    (getTrainUsers())
    //    "../../../result/headers10.csv"
    //    "../../../data/mlboot_data.tsv"
    //    "../../../result/dataset.csv"
    //    writeDataSetRow
    //    true

    // extractTestUsers()



    //let pipeline = LearningPipeline();
    //pipeline.Add(TextLoader("../../../result/dataset30K.csv").CreateFrom(true, ',',false, true, false))
    //pipeline.Add(new StochasticDualCoordinateAscentRegressor())
    //let model = pipeline.Train<Input, Output>()


    //fillDataSet
    //    (getInputs())
    //    "../../../result/headers10.csv"
    //    "../../../result/xusers_data.tsv"
    //    "../../../result/tempResults.csv"
    //    (writePrediction model)
    //    false

    // "000014fe918d1f97a632a796f4948be8" is missing

    extractResults()

    printfn "Hello World from F#!"
    0 // return an integer exit code
