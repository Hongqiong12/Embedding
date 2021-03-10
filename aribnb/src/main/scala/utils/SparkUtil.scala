package utils

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrameReader, SparkSession}

object SparkUtil {
  /**
   * 初始化saprk的一些基础配置
   * @return
   */
  def initSpark(): SparkSession = {

    var sparkBuilder = SparkSession
      .builder()
      .appName(this.getClass.getSimpleName)
    if (new SparkConf()
      .get("spark.master", "NO_MASTER")
      .startsWith("NO_MASTER")) {
      sparkBuilder = sparkBuilder
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "True")
      //        .config("spark.testing.memory", "536870912")
    }
    val spark: SparkSession = sparkBuilder
      .enableHiveSupport()
      .getOrCreate()

    //合并小文件
    spark.conf.set("spark.sql.hive.mergeFiles", "true")
    spark.sparkContext.setLogLevel("ERROR")
    spark
  }

  /**
   * 当需要单独修改某一个配置的时候调用
   *
   * @param confKey:配置的key值
   * @param confVaule：配置的value值
   * @param spark：spark
   */
  private def confSpark(confKey:String,confVaule:String,spark:SparkSession){
    spark.conf.set(confKey,confVaule)
  }


  def initSparkWithConf(confKey:String,confVaule:String): SparkSession ={
    val spark = initSpark()
    confSpark(confKey,confVaule,spark)
    spark
  }

  /**
   * 当单独修改spark的一些配置的时候调用
   * @param confs:很多个spark配置封装的map
   * @return
   */
  def initSparkWithConfs(confs:Map[String,String]): SparkSession ={
    val spark = initSpark()
    for (elem <- confs) {
      confSpark(elem._1,elem._2,spark)
    }
    spark
  }

  /**
   * 根据文件建表
   * @param spark:SparkSession
   * @param path:数据的地址
   * @param tableName：表名
   */
  def createTable(spark:SparkSession, path:String, tableName:String, whichFile:String, seg:String){
    val handelRead: DataFrameReader = spark
      .read
      .option("header", "true")
      .option("delimiter",seg)  //分隔符，默认为 ,
    if(whichFile=="csv")
      handelRead.csv(path).createTempView(tableName)
    else if(whichFile=="parquet") handelRead.parquet(path).createTempView(tableName)
    else if(whichFile=="json") handelRead.json(path).createTempView(tableName)
  }

  /**
   * @return
   */
  def toHaasPath(confPath:String): FileSystem = {
    //以下获取hdfs文件流，然后将模型以流写入hdfs
    val conf = new Configuration
    //相当于通过配置文件的key获取到value的值
    val pathHadoopConf = confPath
    conf.addResource(pathHadoopConf) // 平台上 Haoop_conf 的地址
    /*
   * 更改操作用户有两种方式：（系统会自动识别我们的操作用户，如果我们设置，将会报错会拒绝Administrator用户（windows用户））
   * 1.直接设置运行环境中的用户名为hadoop，此方法不方便因为打成jar包执行还是需要改用户，右键Run As--Run Configurations--Arguments--VM arguments--输入-DHADOOP_USER_NAME=hadoop
   * 2.直接在代码中进行声明
   */
    //更改操作用户为hadoop
    System.setProperty("HADOOP_USER_NAME", "hadoop")
    //获取文件系统对象(目的是获取HDFS文件系统)
    val fs: FileSystem = FileSystem.get(conf)
    fs
  }



}
