// Learn more about F# at http://fsharp.org

open System
open System.IO
open System.Collections.Generic
open Newtonsoft.Json
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers
open Microsoft.ML.Runtime.Api;
open System.Globalization
open Microsoft.ML.Models

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
            if count = 2000
            then goodSet.Add(metricName) |> ignore
            if count > 20000
            then
                goodSet.Remove(metricName) |> ignore
                dict.[metricName] <- -1000000
        else
            dict.[metricName] <- 1)

let loadDataNew() =

    let dict = Dictionary<string, int>()
    let metricDictionary = Dictionary<string, int>(StringComparer.InvariantCultureIgnoreCase)
    let goodMetricSet = HashSet<string>(StringComparer.InvariantCultureIgnoreCase)
    goodMetricSet.Add("label") |> ignore

    let addMetric = addMetricToSet metricDictionary goodMetricSet
    let mutable counter = 0

    let linesWithAnswers  = File.ReadLines("../../../data/mlboot_train_answers_1.tsv")
    for line in linesWithAnswers do
        let values = line.Split('\t')
        dict.[values.[0]] <- values.[1] |> int

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

let getUsers path =
    let dict = Dictionary<string, int>()
    let linesWithAnswers  = File.ReadLines(path)
    for line in linesWithAnswers do
        let values = line.Split('\t')
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

    let userValues = Dictionary<string, single>(StringComparer.InvariantCultureIgnoreCase)
    let userValuesCounter = Dictionary<string, (int*int)>()
    headers |> Seq.iter (fun elem -> userValues.Add(elem, 0.0f))

    let mutable currentId = ""

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
                        userValues.[kv.Key] <- (sum |> single) / (count |> single)

                f currentId userValues sr
                counter <- counter + 1
                if counter % 100 = 0
                then printfn "%A done" counter
                userValues.Clear()
                userValuesCounter.Clear()
                headers |> Seq.iter (fun elem -> userValues.Add(elem, 0.0f))

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
            userValues.["cat" + category.ToString()] <- 1.0f
            userValues.["label"] <- dict.[id] |> single

let extractTestUsers resultPath =
    let dict = Dictionary<string, int>()
    let linesWithAnswers  = File.ReadLines("../../../data/mlboot_test.tsv")
    for line in linesWithAnswers do
        let values = line.Split('\t')
        if
            values.[0] <> "cuid"
        then
            dict.[values.[0]] <- 1

    let linesWithData  = File.ReadLines("../../../data/mlboot_data.tsv")
    use fs = File.OpenWrite(resultPath)
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
        [<VectorType(20240)>]
        val mutable Features: single[]

        new() = { Label = 0.0f; Features = [||] }


type Output =

        [<ColumnName("Score")>]
        val mutable Score: single

        new() = { Score = 0.0f; }


let writeDataSetRow id (userValues: Dictionary<string, single>) (sr: StreamWriter) =
    let str = String.Join(',', userValues.Values |> Seq.map (fun value -> value.ToString(CultureInfo.InvariantCulture)))
    sr.WriteLine(str)

let writePrediction (model: PredictionModel<Input, Output>) id (userValues: Dictionary<string, single>) (sr: StreamWriter) =
    let label = userValues.["label"] |> single
    userValues.Remove("label") |> ignore
    let input = new Input()
    input.Label <- label
    input.Features <- userValues.Values |> Seq.toArray
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

let evaluateResults (model: PredictionModel<Input, Output>) (evaluatePath: string) =
    let testData = TextLoader(evaluatePath).CreateFrom(true, ',',false, true, false)
    //let regressionEvaluator = new RegressionEvaluator()
    //let metrics = regressionEvaluator.Evaluate(model, testData)
    //Console.WriteLine()
    //Console.WriteLine("PredictionModel quality metrics evaluation")
    //Console.WriteLine("------------------------------------------")
    //Console.WriteLine("Rms = {0}", metrics.Rms)
    //Console.WriteLine("RSquared = {0}", metrics.RSquared)
    let regressionEvaluator = new BinaryClassificationEvaluator ()
    let metrics = regressionEvaluator.Evaluate(model, testData)
    Console.WriteLine()
    Console.WriteLine("PredictionModel quality metrics evaluation")
    Console.WriteLine("------------------------------------------")
    Console.WriteLine("Accuracy: {0}", metrics.Accuracy)
    Console.WriteLine("Auc: {0}", metrics.Auc)
    Console.WriteLine("F1Score: {0}", metrics.F1Score)

[<EntryPoint>]
let main argv =

    //let headers = loadDataNew()
    //File.WriteAllText("../../../result/headers(2-20).csv", String.Join(",",headers))

    //fillDataSet
    //    (getUsers "../../../data/mlboot_train_answers_1.tsv")
    //    "../../../result/headers(2-20).csv"
    //    "../../../data/mlboot_data.tsv"
    //    "../../../result/dataset_train.csv"
    //    writeDataSetRow
    //    true

    //fillDataSet
    //    (getUsers "../../../data/mlboot_train_answers_2.tsv")
    //    "../../../result/headers(2-20).csv"
    //    "../../../data/mlboot_data.tsv"
    //    "../../../result/dataset_evaluate.csv"
    //    writeDataSetRow
    //    true

    // extractTestUsers "../../../data/test_users_data.tsv"


    let pipeline = LearningPipeline();
    pipeline.Add(TextLoader("../../../result/dataset_train.csv").CreateFrom(true, ',',false, true, false))
    pipeline.Add(new FieldAwareFactorizationMachineBinaryClassifier ())
    let model = pipeline.Train<Input, Output>()
    evaluateResults model "../../../result/dataset_evaluate.csv"

    model.WriteAsync("../../../result/model.zip").Wait()

    //let classifier = new FieldAwareFactorizationMachineBinaryClassifier()
    //classifier.GetInputData <-

    //fillDataSet
    //    (getInputs())
    //    "../../../result/headers10.csv"
    //    "../../../data/test_users_data.tsv"
    //    "../../../result/tempResults.csv"
    //    (writePrediction model)
    //    false

    // "000014fe918d1f97a632a796f4948be8" is missing

    //extractResults()

    printfn "Hello World from F#!"
    0 // return an integer exit code
