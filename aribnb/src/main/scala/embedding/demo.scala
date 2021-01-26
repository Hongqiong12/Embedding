package embedding

import org.apache.spark.ml.feature.{Word2VecOptiModel, Word2VecOptis}
import org.apache.spark.mllib.feature.GloveModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import utils.SparkUtil

import java.net.URL
import scala.collection.mutable

object demo {

  def main(args: Array[String]): Unit = {
    // 初始化spark
    val spark = SparkUtil.initSpark()
    spark.sparkContext.setLogLevel("Error")
    import spark.implicits._
    // 读取sentence
    val rootPath= this.getClass.getClassLoader.getResource("")
    val walkSeqDf: DataFrame = spark
      .read
      .textFile(s"$rootPath/part-00000")
      .toDF("walkSeq")
    // 爱彼迎的论文复现，将ORDER的商品作为全局的context
//    word2vec(spark, rootPath, walkSeqDf)
    // GLOVE分布式实现的DEMO
    glove(spark, walkSeqDf)

  }
  private def glove(spark: SparkSession, walkSeqDf: DataFrame): DataFrame = {
    val value = walkSeqDf.rdd.map(x => x.getString(0).split(","))
    val gloveModel = new GloveModel()
    val value1 = gloveModel.fit(value)
    //创建schema信息
    val structSchema: StructType = StructType(
      List(

        StructField("word"     , StringType, true),
        StructField("embedding", StringType, true)
      )
    )
    val wordVecDf = spark.createDataFrame(value1.map(x=>Row(x._1, x._2)),structSchema)
    wordVecDf.show(10,false)
    wordVecDf
  }


  private def word2vec(spark: SparkSession, rootPath: URL, walkSeqDf: DataFrame) = {
    import spark.implicits._
    val word2Model: Word2VecOptiModel = new Word2VecOptis() // todo skip-gram model
      .setInputCol("walkSeq")
      .setOutputCol("result")
      .setVectorSize(32)
      .setWindowSize(5)
      .setMinCount(0) //大于25个的词数量：4639750    大于45个的词数量：3210577
      .setMaxIter(1)
      .setStepSize(0.01)
      .setEOS("-2") // 尾部终止符号，用来判断最后一个商品是否是点击的商品
      .fit(walkSeqDf.map(x=>x.getString(0).split(",")))
    word2Model.write.overwrite().save(s"$rootPath/modelFile")
    // 读取生成的embedding 向量
    val resEmbed = spark.read.parquet(s"$rootPath/modelFile/data/")
      .rdd
      .map { x =>
        val skuId: String = x.getString(0)
        val vector: Array[Float] = x.getAs[mutable.WrappedArray[Float]](1).toArray.map(x => x)
        (skuId, vector.mkString(","))
      }.toDF("skuId", "vector")
    //    resEmbed.show(100, false)
    // 其中假设9435_422036 是下单的商品
    resEmbed.where("skuId='9964_605249' or skuId='16796_348014'")
      .show(2, false)
  }
}
