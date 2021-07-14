package com.intel.analytics.zoo.serving

import com.intel.analytics.zoo.pipeline.api.net.TFNet
import com.intel.analytics.zoo.serving.http.ServingFrontendSerializer
import com.intel.analytics.zoo.serving.utils.Supportive
import org.scalameter.{Key, Measurer, Warmer, config}
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

object ArrayDien extends Supportive{
  case class Params(modelPath: String = "", dataPath: String = "")
  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[Params]("DIEN") {
      opt[String]('m', "modelPath")
        .text("model path")
        .action((x, params) => params.copy(modelPath = x))
      opt[String]('d', "dataPath")
        .text("data Path")
        .action((x, params) => params.copy(dataPath = x))
    }
    val arg = parser.parse(args, Params()).head
    val timeArray = new ArrayBuffer[String]()

    val model = TFNet(arg.modelPath)
    val inputStr = scala.io.Source.fromFile(arg.dataPath).mkString
    val input = ServingFrontendSerializer.deserialize(inputStr)

    (0 until 200).foreach(i => {
      timing("preprocessing") {
        var x = 0
        for (i <- 1 to 300000) {
          x += 1
        }
      }
      timing ("predict") {
        model.updateOutput(input)
      }
    })

//      val modelArray = new Array[TFNet](3)
//      (0 until 3).foreach(i => modelArray(i) = TFNet(arg.modelPath))
//
//      (0 until 1).indices.toParArray.map(thrd => {
//        val inputStr = scala.io.Source.fromFile(arg.dataPath).mkString
//        val input = ServingFrontendSerializer.deserialize(inputStr)
////        val time = config(
////          Key.exec.benchRuns -> 100,
////          Key.verbose -> true
////        ) withWarmer {
////          new Warmer.Default
////        } withMeasurer {
////          new Measurer.IgnoringGC
////        } measure {
////          modelArray(thrd).updateOutput(input)
////        }
//        (0 until 200).foreach(i => {
//          timing("preprocessing") {
//            var x = 0
//            for (i <- 1 to 300000) {
//              x += 1
//            }
//          }
//          timing ("predict") {
//            modelArray((thrd) % 3).updateOutput(input)
//          }
//
//        })
//      })

  }
}

