// Learn more about F# at http://fsharp.org

open System
open System.IO
open System.Collections.Generic
open Newtonsoft.Json

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

let appendDictionary (main: Dictionary<int, (byte*byte)>) (adding: Dictionary<int, byte>) = 
    for elem in adding do
        if (main.ContainsKey elem.Key)
        then 
            let (sum, num) = main.[elem.Key]
            main.[elem.Key] <- (sum + elem.Value, num + 1uy)
        else
            main.[elem.Key] <- (elem.Value, 1uy)

[<EntryPoint>]
let main argv =
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


    printfn "Hello World from F#!"
    0 // return an integer exit code
