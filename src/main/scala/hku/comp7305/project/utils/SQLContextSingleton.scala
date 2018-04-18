package hku.comp7305.project.utils

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}

object SQLContextSingleton {

  @transient
  @volatile private var instance: SQLContext = _

  def getInstance(spark: SparkContext): SQLContext = {
    if (instance == null) {
      synchronized {
        if (instance == null) {
//          instance = spark.sqlContext
          instance = SQLContext.getOrCreate(spark)
        }
      }
    }
    instance
  }
}
