package hku.comp7305.project

import hku.comp7305.project.SVMModelCreator.{createAndSaveModel, loadStopWords, validateAccuracyOfModel}
import hku.comp7305.project.utils.{LogUtil, PropertiesLoader}
import org.apache.spark.sql.SparkSession

/**
 * java -cp target/TwitterSentimentAnalysis-1.0.jar hku.comp7305.project.App
 * java -cp target/TwitterSentimentAnalysis-1.0.jar hku.comp7305.project.NaiveBayesModelCreator
 */
object App {
  def main(args: Array[String]): Unit = {

    if (args.length < 1) {
      LogUtil.error("Need indicate 'train' or 'show' as argument!")
      return
    }

    var trainMode = true
    var iterations = 0
    if (args(0).toString.trim.equals("train")) {
      trainMode = true
      if (args.length != 2) {
        LogUtil.error("SVM need iteration number!")
        return
      } else {
        try {
          iterations = args(1).toString.toInt
        } catch {
          case e:Exception => println(e)
            LogUtil.error("Invalid iteration number!")
            return
        }
      }
    }  else if (args(0).toString.trim.equals("show")) {
      LogUtil.warn("Not implemented yet!")
      trainMode = false
      return
    } else {
      LogUtil.error("Only 'train' or 'show' are valid arguments!")
      return
    }

    val spark = SparkSession.builder
                    .appName("Twitter Movie Reviews Sentiment Analysis v0.3")
                    .getOrCreate()
    val sc = spark.sparkContext
//    println("PropertiesLoader.SPARK_LOG_LEVEL: " + PropertiesLoader.SPARK_LOG_LEVEL)
//    sc.setLogLevel(PropertiesLoader.SPARK_LOG_LEVEL)

    if (trainMode) {
      LogUtil.info("Starting training model...")
      val stopWordsList = sc.broadcast(loadStopWords(sc, PropertiesLoader.NLTK_STOPWORDS_PATH))
      createAndSaveModel(sc, stopWordsList, iterations)
      LogUtil.info("Validating  model...")
      validateAccuracyOfModel(sc, stopWordsList)
    } else {
      LogUtil.warn("Not implemented yet!")
      //      println("Not implemented yet!")
    }
  }
}
