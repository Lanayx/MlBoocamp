// Learn more about F# at http://fsharp.org

open System
open System.IO
open System.Collections.Generic
open Newtonsoft.Json
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Data

type UserData =
    {
        // Category: byte
        Metrics: Dictionary<int, (byte*byte)>
        // Days: byte
        Vote: byte
    }

let deserializeMetric metric =
    let obj = JsonConvert.DeserializeObject<Dictionary<int, byte>>(metric)
    obj

let getMetric metric =
    let obj = JsonConvert.DeserializeObject<Dictionary<string, byte>>(metric)
    obj

let appendDictionary (main: Dictionary<int, (byte*byte)>) (adding: Dictionary<int, byte>) =
    for elem in adding do
        if (main.ContainsKey elem.Key)
        then
            let (sum, num) = main.[elem.Key]
            main.[elem.Key] <- (sum + elem.Value, num + 1uy)
        else
            main.[elem.Key] <- (elem.Value, 1uy)

let loadData () =
    let dict = Dictionary<string, byte>()
    let metrics = Dictionary<string, UserData>()
    let linesWithAnswers  = File.ReadLines("../../../data/mlboot_train_answers.tsv")
    for line in linesWithAnswers do
        let values = line.Split('\t')
        if
            values.[0] <> "cuid"
        then
            dict.[values.[0]] <- values.[1] |> byte

    let linesWithData  = File.ReadLines("../../../data/mlboot_data.tsv")
    for line in linesWithData do
        let values = line.Split('\t')
        let id = values.[0]
        if
            dict.ContainsKey(id)
        then
            let metrics2 = deserializeMetric values.[2]
            let metrics3 = deserializeMetric values.[3]
            let metrics4 = deserializeMetric values.[4]
            if metrics.ContainsKey(id)
            then
                let metricsDict = metrics.[id].Metrics
                metrics2 |> appendDictionary metricsDict
                metrics3 |> appendDictionary metricsDict
                metrics4 |> appendDictionary metricsDict
            else

                let metricsDist = Dictionary<int, (byte*byte)>()
                metrics2 |> appendDictionary metricsDist
                metrics3 |> appendDictionary metricsDist
                metrics4 |> appendDictionary metricsDist
                metrics.[id] <-
                                {
                                    Metrics = metricsDist
                                    Vote = dict.[id]
                                }
    metrics

let extractMetrics (metric: Dictionary<string, byte>) =
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
            if count = 1000
            then goodSet.Add(metricName) |> ignore
        else
            dict.[metricName] <- 1)

let loadDataNew() =

    let dict = Dictionary<string, byte>()
    let metricDictionary = Dictionary<string, int>(StringComparer.InvariantCultureIgnoreCase)
    let goodMetricSet = HashSet<string>(StringComparer.InvariantCultureIgnoreCase)

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
    //let results = seq {
    for line in linesWithData do
        counter <- counter + 1
        let res = line.Split('\t')
        let [|id; category; metric1; metric2; metric3; days |] = res
        if dict.ContainsKey(id)
        then
            addMetric "x" metric1
            addMetric "y" metric2
            addMetric "z" metric3
    //}
    [0..5] |> List.iter (fun i -> goodMetricSet.Add("cat"+i.ToString()) |> ignore)
    goodMetricSet.Add("days") |> ignore
    goodMetricSet.Add("id") |> ignore
    goodMetricSet.Add("result") |> ignore
    goodMetricSet

[<EntryPoint>]
let main argv =

    // let headers = loadDataNew()

    // File.WriteAllText("../../../result/headers.csv", String.Join(",",headers))

    //let customLoader = TextLoader("abcd")
    //customLoader.

    let headers = File.ReadAllText("../../../result/headers.csv")
    let values = headers.Split(',')
    let array = [1..values.Length]
    File.WriteAllLines("../../../result/headerswork.csv", [| headers; String.Join(',',array) |])

    let pipeline = LearningPipeline();
    pipeline.Add(TextLoader("../../../result/headerswork.csv").CreateFrom(true, ',',false, true, false))

    printfn "Hello World from F#!"
    0 // return an integer exit code
