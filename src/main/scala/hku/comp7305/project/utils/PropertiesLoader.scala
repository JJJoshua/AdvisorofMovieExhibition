package hku.comp7305.project.utils

import java.io.FileInputStream
import java.util.Properties


object PropertiesLoader {

  val DEFAULT_SENTIMENT140_TRAIN_DATA_PATH = "hdfs:///tsa/trainingandtestdata/training.1600000.processed.noemoticon.csv"
  val DEFAULT_SENTIMENT140_TEST_DATA_PATH = "hdfs:///tsa/trainingandtestdata/testdata.manual.2009.06.14.csv"
  val DEFAULT_MODEL_PATH = "hdfs:///tsa/model"
  val DEFAULT_NLTK_STOPWORDS_PATH = "hdfs:///tsa/trainingandtestdata/NLTK_English_Stopwords_Corpus.txt"
  val DEFAULT_TEST_DATA_PATH = "hdfs:///data"
  val DEFAULT_PROCESSED_TWEETS_PATH = "hdfs:///processed_data"

  private val props = new Properties()
  props.load(new FileInputStream("application.conf"))

  val SENTIMENT140_TRAIN_DATA_PATH = props.getProperty("SENTIMENT140_TRAIN_DATA_PATH", DEFAULT_SENTIMENT140_TRAIN_DATA_PATH)
  val SENTIMENT140_TEST_DATA_PATH = props.getProperty("SENTIMENT140_TEST_DATA_PATH", DEFAULT_SENTIMENT140_TEST_DATA_PATH)
  val MODEL_PATH = props.getProperty("MODEL_PATH", DEFAULT_MODEL_PATH)
  val NLTK_STOPWORDS_PATH = props.getProperty("NLTK_STOPWORDS_PATH", DEFAULT_NLTK_STOPWORDS_PATH)
  val TEST_DATA_PATH = props.getProperty("TEST_DATA_PATH", DEFAULT_TEST_DATA_PATH)
  val PROCESSED_TWEETS_PATH = props.getProperty("PROCESSED_TWEETS_PATH", DEFAULT_PROCESSED_TWEETS_PATH)
}
