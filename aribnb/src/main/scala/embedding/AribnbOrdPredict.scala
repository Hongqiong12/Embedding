package embedding

import org.apache.spark.ml.feature.{Word2VecOptiModel, Word2VecOptis}
import utils.SparkUtil

import scala.collection.mutable

object AribnbOrdPredict {

  def main(args: Array[String]): Unit = {
    // 初始化spark
    val spark = SparkUtil.initSpark()
    spark.sparkContext.setLogLevel("Error")
    import spark.implicits._
    // 读取sentence
    val rootPath= this.getClass.getClassLoader.getResource("")
    val walkSeqDf = spark
      .read
      .textFile(s"$rootPath/part-00000")
      .map(x=>x.split(","))
      .toDF("walkSeq")
    val word2Model: Word2VecOptiModel = new Word2VecOptis() // todo skip-gram model
      .setInputCol("walkSeq")
      .setOutputCol("result")
      .setVectorSize(32)
      .setWindowSize(5)
      .setMinCount(0) //大于25个的词数量：4639750    大于45个的词数量：3210577
      .setMaxIter(1)
      .setStepSize(0.01)
      .fit(walkSeqDf)
    word2Model.write.overwrite().save(s"$rootPath/modelFile")
    // 读取生成的embedding 向量
    val resEmbed = spark.read.parquet(s"$rootPath/modelFile/data/")
      .rdd
      .map { x =>
        val skuId: String = x.getString(0)
        val vector: Array[Float] = x.getAs[mutable.WrappedArray[Float]](1).toArray.map(x=>x)
        (skuId, vector.mkString(","))
      }.toDF("skuId", "vector")
//    resEmbed.show(100, false)
    // 其中假设9435_422036 是下单的商品
    resEmbed.where("skuId='9964_605249' or skuId='16796_348014'")
      .show(2,false)


  }


}
