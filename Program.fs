// Learn more about F# at http://fsharp.org

open System
open System.IO
open System.Collections.Generic
open Newtonsoft.Json
open Microsoft.ML
open Microsoft.ML.Data

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

let fillDataSet headersPath =
    let mutable counter = 0
    let dict = Dictionary<string, int>()
    let linesWithAnswers  = File.ReadLines("../../../data/mlboot_train_answers.tsv")
    for line in linesWithAnswers do
        let values = line.Split('\t')
        if
            values.[0] <> "cuid"
        then
            dict.[values.[0]] <- values.[1] |> int

    let headersString = File.ReadAllText(headersPath)
    let headers = headersString.Split(',')

    let linesWithData  = File.ReadLines("../../../data/mlboot_data.tsv")
    File.WriteAllText("../../../result/dataset.csv", String.Empty);
    use fs = File.OpenWrite("../../../result/dataset.csv")
    use sr = new StreamWriter(fs)
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
                    let str = String.Join(',', userValues.Values)
                    sr.WriteLine(str)
                    counter <- counter + 1
                    if (counter % 100) = 0
                    then sr.Flush()
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


    sr.Flush()
    fs.Close()

[<EntryPoint>]
let main argv =

    //let headers = loadDataNew()
    //File.WriteAllText("../../../result/headers10.csv", String.Join(",",headers))

    //let array = [1..values.Length]
    //File.WriteAllLines("../../../result/headerswork.csv", [| headers; String.Join(',',array) |])

    fillDataSet "../../../result/headers10.csv"

    let pipeline = LearningPipeline();
    pipeline.Add(TextLoader("../../../result/headerswork2.csv").CreateFrom(true, ',',false, true, false))

    printfn "Hello World from F#!"
    0 // return an integer exit code
