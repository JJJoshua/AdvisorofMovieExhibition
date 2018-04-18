package hku.comp7305.project

import hku.comp7305.project.SVMModelCreator.loadStopWords
import hku.comp7305.project.utils.{Constants, LogUtil, PropertiesLoader}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession
import org.elasticsearch.spark._
import org.elasticsearch.spark.rdd.EsSpark

import scala.util.Try

object ProcessData {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Twitter Movie Reviews Sentiment Analysis v0.3 (Process Data)")
      .getOrCreate()
    val sc = spark.sparkContext
    LogUtil.info("Starting processing...")
    val stopWordsList = sc.broadcast(loadStopWords(sc, PropertiesLoader.NLTK_STOPWORDS_PATH))
    processBySVM(sc, stopWordsList)
    LogUtil.info("Finished processing...")
  }

  def processBySVM(sc: SparkContext, stopWordsList: Broadcast[List[String]]) = {
      process(sc, PropertiesLoader.TEST_DATA_PATH, stopWordsList)
  }

  def process(sc: SparkContext, pathName: String, stopWordsList: Broadcast[List[String]]) = {
    val hadoopConf = sc.hadoopConfiguration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val path = new Path(pathName)

    import org.apache.hadoop.fs.FileUtil
    import org.json4s._
    import org.json4s.jackson.JsonMethods._

    //    case class Tweet(nbr_retweet:Int,
    //                      nbr_favorite:Int,
    //                      user_id:String,
    //                      url:String,
    //                      text:String,
    //                      usernameTweet:String,
    //                      datetime:String,
    //                      is_retweet:Boolean,
    //                      ID:String,
    //                      nbr_reply:Int,
    //                      is_reply:Boolean
    //    )
    //
    //    case class Tweet(text:String)
    //
    //    implicit val formats = Serialization.formats(NoTypeHints)
    //    val str = """{"nbr_retweet": 0,"nbr_favorite": 0,"user_id": "3247178202","url": "/MoAMPmusic/status/980664030061834240","text": "Everyone Wants to be a Lion. \nChances are most don't get the opportunity to be. \n#zoology","usernameTweet": "MoAMPmusic","datetime": "2018-04-02 12:31:39","is_retweet": false,"ID": "980664030061834240","nbr_reply": 0,"is_reply": false},"""
    //    val tweetObj = parse(str).extract[Tweet]
    //    parse(str).values

    /*
    {"nbr_retweet": 0,
    "nbr_favorite": 0,
    "user_id": "3247178202",
    "url": "/MoAMPmusic/status/980664030061834240",
    "text": "Everyone Wants to be a Lion. \nChances are most don't get the opportunity to be. \n#zoology",
    "usernameTweet": "MoAMPmusic",
    "datetime": "2018-04-02 12:31:39",
    "is_retweet": false,
    "ID": "980664030061834240",
    "nbr_reply": 0,
    "is_reply": false},
     */
    case class TweetES(city_name:String, genre:String, movie_name:String, sentiment:String, location:String, time:String)
    val model = SVMModelCreator.loadModel(sc)

    val cityPath = hdfs.listStatus(path)
    val cities = FileUtil.stat2Paths(cityPath)
    var cityCount = 0
    for (city <- cities) {
//      println("city: " + city.getName)
      cityCount += 1
      val cityName = city.getName
      val genrePath = hdfs.listStatus(city)
      val genres = FileUtil.stat2Paths(genrePath)

      var genreCount = 0
      for (genre <- genres) {
        genreCount += 1
//        println("genre: " + genre.getName)
        LogUtil.info("\n\n\t[=============> " + city.getName + "(" + cityCount + "/" + cities.length + ")" + " - " + genre.getName +  "(" + genreCount + "/" + genres.length + ") <==============]\n")
        val genreName = genre.getName
        val movies = sc.wholeTextFiles(genre.toString, 20) // 8
        val cleanedMovies = movies.filter(
          x => {
            val pathSplits = x._1.split("/")
            val fileName = pathSplits(pathSplits.length - 1)
            val splits = fileName.split('-')
            splits.length == 3
          }
        )
        val moviesTweets = movies.flatMap {
          case (pathName:String, tweets:String) => {
            try {
              val pathSplits = pathName.split("/")
              val fileName = pathSplits(pathSplits.length - 1)
              val splits = fileName.split('-')
              val movieName = splits(0).substring(0, splits(0).length - "near".length).replaceAll("_", " ").trim.replaceAll("#", " ").trim
              tweets.split("\n").map(
                t => {
                  try {
                    implicit val formats = DefaultFormats
                    val text: String = (parse(t) \ "text").extract[String]
                    val time: String = (parse(t) \ "datetime").extract[String]

                    // ===
                    val sentimentFLoat = SVMModelCreator.predict(model, text, stopWordsList)
                    var sentiment = "pos"
                    if (sentimentFLoat == 0.0) {
                      sentiment = "neg"
                    }
                    // ===
                    // case class TweetES(city_name:String, genre:String, movie_name:String, sentiment:String, location:String, time:String)
                    TweetES(cityName, genreName, movieName, sentiment, Constants.geoMap(cityName.trim), time)
                  } catch {
                    case e:Exception =>
                      LogUtil.warn("Exception when convert raw text: " + t + "\n\n Get exception: " + e.toString)
                      None
                  }
                }
              )
            } catch {
              case e:Exception =>
                LogUtil.warn("Exception : " + e.toString)
                None
            }
          }
        }
//        println(moviesTweets.collect().toList.toString())
        val hdfsSavePath = PropertiesLoader.PROCESSED_TWEETS_PATH + "/" + cityName.replaceAll(" ", "_") + "-" + genre.getName.replaceAll(" ", "_") + ".data"
        SVMModelCreator.checkModelSavePath(sc, hdfsSavePath)
        LogUtil.info("\n\n\t ==>  Save data to path: " + hdfsSavePath + "\n")
//        moviesTweets.saveAsTextFile(hdfsSavePath)
        EsSpark.saveToEs(moviesTweets, "tweets/tweet")
      }
    }
  }

  def extractInfoFromFileName(path: String): String = {
    //     hdfs://student10:9000/test/Chicago/drama/#Zoology_near-"Chicago"_since-2016
    //    val fileName = new Path(path).getName
    val pathSplits = path.split("/")
    val fileName = pathSplits(pathSplits.length - 1)
    val splits = fileName.split('-')
    if (splits.length != 3) {
      println("ERROR: invalid file " + fileName)
      null
    } else {
      val movieName = splits(0).substring(0, splits(0).length - "near".length).replaceAll("_", " ").trim.replaceAll("#", " ").trim
      //      val sinceYear = splits(2).toInt
      movieName
    }
  }
}
