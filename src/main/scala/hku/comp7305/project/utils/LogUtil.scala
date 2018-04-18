package hku.comp7305.project.utils

import org.apache.log4j.Logger

object LogUtil {

  val logger = Logger.getLogger("Twitter Movie Reviews Sentiment Analysis")
  def info(msg: String):Unit = {
    logger.info(msg)
  }

  def warn(msg: String):Unit = {
    logger.warn(msg)
  }

  def error(msg: String):Unit = {
    logger.error(msg)
  }
}
