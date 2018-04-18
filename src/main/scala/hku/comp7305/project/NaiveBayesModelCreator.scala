package hku.comp7305.project

import java.io.IOException

import hku.comp7305.project.utils.{LogUtil, PropertiesLoader, SQLContextSingleton}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.mllib.linalg.Vector

object NaiveBayesModelCreator {

  def main(args: Array[String]): Unit = {
//    val spark = SparkSession.builder
//      .appName("Simple Application")
//      .master("local")
//      .getOrCreate()
//    val sc = spark.sparkContext
//    val stopWordsList = sc.broadcast(loadStopWords(spark.sparkContext, PropertiesLoader.NLTK_STOPWORDS_PATH))
//    createAndSaveModel(sc, stopWordsList)
//    validateAccuracyOfModel(sc, stopWordsList)
  }

  val hashingTF = new HashingTF()

  def createAndSaveModel(sc: SparkContext, stopWordsList: Broadcast[List[String]], iterations: Int): Unit = {
    LogUtil.info("Loading training file...")
    val tweetsDF: DataFrame = loadSentiment140File(sc, PropertiesLoader.SENTIMENT140_TRAIN_DATA_PATH)
    val labeledRDD = tweetsDF.select("polarity", "text").rdd.map {
      case Row(polarity: Int, tweet: String) =>
        val tweetInWords: Seq[String] = getCleanedTweetText(tweet, stopWordsList.value)
        LabeledPoint(polarity, textToFeatureVector(tweetInWords))
    }
    labeledRDD.cache()
    LogUtil.info("Starting training Naive Bayes model...")
    val model: NaiveBayesModel = NaiveBayes.train(labeledRDD, lambda = 0.1, modelType = "multinomial")
//    val model = SVMWithSGD.train(labeledRDD, iterations)
    MultilayerPerceptronClassificationModel
    //TODO
    LogUtil.info("Training Naive Bayes model finished!")
    LogUtil.info("Saving Naive Bayes model...")
    checkModelSavePath(sc, PropertiesLoader.MODEL_PATH)
    model.save(sc, PropertiesLoader.MODEL_PATH)
    LogUtil.info("Saving Naive Bayes model finished!")
  }

  def checkModelSavePath(sc: SparkContext, pathName: String): Unit = {
    val hadoopConf = sc.hadoopConfiguration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val path = new Path(pathName)
    if (hdfs.isDirectory(path)) {
        hdfs.delete(path, true)
    }
  }

  def replaceNewLines(tweet: String) = {
    tweet.replaceAll("\n", "")
  }

  def validateAccuracyOfModel(sc: SparkContext, stopWordsList: Broadcast[List[String]]): Unit = {
    val model: NaiveBayesModel = NaiveBayesModel.load(sc, PropertiesLoader.MODEL_PATH)
//    val model = SVMModel.load(sc, PropertiesLoader.MODEL_PATH)
    //TODO
    val tweetsDF: DataFrame = loadSentiment140File(sc, PropertiesLoader.SENTIMENT140_TEST_DATA_PATH)
    val actualVsPredictionRDD = tweetsDF.select("polarity", "text").rdd.map {
      case Row(polarity: Int, tweet: String) =>
        val tweetText = replaceNewLines(tweet)
        val tweetInWords: Seq[String] = getCleanedTweetText(tweetText, stopWordsList.value)
        (polarity.toDouble,
          model.predict(textToFeatureVector(tweetInWords)),
          tweetText)
    }
    val accuracy = 100.0 * actualVsPredictionRDD.filter(x => x._1 == x._2).count() / tweetsDF.count()
    /*actualVsPredictionRDD.cache()
    val predictedCorrect = actualVsPredictionRDD.filter(x => x._1 == x._2).count()
    val predictedInCorrect = actualVsPredictionRDD.filter(x => x._1 != x._2).count()
    val accuracy = 100.0 * predictedCorrect.toDouble / (predictedCorrect + predictedInCorrect).toDouble*/
    println(f"""\n\t<==******** Prediction accuracy compared to actual: $accuracy%.2f%% ********==>\n""")
//    saveAccuracy(sc, actualVsPredictionRDD)
  }

  def loadSentiment140File(sc: SparkContext, sentiment140FilePath: String): DataFrame = {
    val sqlContext = SQLContextSingleton.getInstance(sc)
    val tweetsDF = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .option("inferSchema", "true")
      .load(sentiment140FilePath)
      .toDF("polarity", "id", "date", "query", "user", "text")
    // Drop the columns we are not interested in.
    tweetsDF.drop("id").drop("date").drop("query").drop("user")
  }

  def getCleanedTweetText(tweetText: String, stopWordsList: List[String]): Seq[String] = {
    //Remove URLs, RT, MT and other redundant chars / strings from the tweets.
    tweetText.toLowerCase()
      .replaceAll("\n", "")
      .replaceAll("rt\\s+", "")
      .replaceAll("\\s+@\\w+", "")
      .replaceAll("@\\w+", "")
      .replaceAll("\\s+#\\w+", "")
      .replaceAll("#\\w+", "")
      .replaceAll("(?:https?|http?)://[\\w/%.-]+", "")
      .replaceAll("(?:https?|http?)://[\\w/%.-]+\\s+", "")
      .replaceAll("(?:https?|http?)//[\\w/%.-]+\\s+", "")
      .replaceAll("(?:https?|http?)//[\\w/%.-]+", "")
      .split("\\W+")
      .filter(_.matches("^[a-zA-Z]+$"))
      .filter(!stopWordsList.contains(_))
  }

  def textToFeatureVector(tweetText: Seq[String]): Vector = {
    hashingTF.transform(tweetText)
  }

  def loadStopWords(sc: SparkContext, stopWordsFileName: String): List[String] = {
//    Source.fromInputStream(getClass.getResourceAsStream(stopWordsFileName)).getLines().toList
//    Source.fromInputStream(new FileInputStream(stopWordsFileName)).getLines().toList
    sc.textFile(stopWordsFileName).collect().toList
  }
}
