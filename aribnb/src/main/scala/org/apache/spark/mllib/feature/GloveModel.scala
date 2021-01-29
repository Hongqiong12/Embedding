/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feature

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import java.util.Date
import scala.collection.mutable

/**
 * GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
 * Training is performed on aggregated global word-word co-occurrence statistics from a corpus,
 * and the resulting representations showcase interesting linear substructures
 * of the word vector space.
 * Original C implementation, and research paper http://www-nlp.stanford.edu/projects/glove/
 * @param window number of context words from [-window, window]
 * @param numComponents number of latent dimensions
 * @param minCount minimum frequency to consider a vocabulary word
 * @param learningRate learning rate for SGD estimation.
 * @param alpha, maxCount weighting function parameters, not extremely sensitive to corpus,
 *        though may need adjustment for very small or very large corpora
 */
class GloveModel  (
                    window:        Int    = 5,
                    numComponents: Int    = 50,
                    minCount:      Int    = 0,
                    learningRate:  Double = 0.05d,
                    alpha:         Double = 0.75d,
                    seed:          Long   = 2L ,
                    vectorSize:    Int    = 32,
                    numIterations: Int    = 1
                  ) extends Serializable with Logging {

  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var vocabSize: Int = 1
  private var vocab: Array[(String, Int)] = new Array[(String, Int)](vocabSize.toInt)
  private var maxCount = 100.0d


  // 计算序列的一些基础信息，词典的个数及每个词典中词的编号
  private def learnVocab(dataset: RDD[Array[String]]) = {
    val words: RDD[String] = dataset.flatMap(x => x)
    // 单词及其对应的频次，此处就涉及到内存的问题。此处的代码参考word2vec
    vocab= words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .collect()
      .sortBy(_._2)(Ordering[Int].reverse)
    //得到最大词的个数
    maxCount = vocab(vocab.length-1)._2.toDouble
    // 即可求得所有词的个数
    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")
    // 按照词频从大到小进行编号
    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a)._1 -> a
      a += 1
    }
    println(s"------总的单词个数为：$vocabSize-----")
  }



  def fit(dataset: RDD[Array[String]]): RDD[(String, String)] = {
    val sc = dataset.context
    learnVocab(dataset)
    val coMatrix: RDD[(Int, Int, Double)] = coocurrenceMatrix(dataset)
    val bcVocabHash = sc.broadcast(vocabHash)
    val value = doFit(dataset, bcVocabHash, coMatrix)
    value
  }

  //计算共现矩阵，输出为(开始点，终止点，共现次数)，注意为了防止重复计算，因为 ab和ba在glove中无明显差异，因此约定编号小的在左侧
  private def coocurrenceMatrix(dataset: RDD[Array[String]]): RDD[(Int, Int, Double)] ={
    val sc = dataset.context
    val bcVocabHash = sc.broadcast(vocabHash)
    val coocurrenceMatrix: RDD[Iterator[(Int, Int, Double)]] = dataset
      .map{iter: Array[String] =>
      {
        var coocurences = scala.collection.mutable.HashMap.empty[(Int, Int), Double]
        var windowBuffer = List.empty[Int]
        iter.foreach { w =>
          val word = bcVocabHash.value.get(w).map(w => {
            for {
              (contextWord, i) <- windowBuffer.zipWithIndex
              if (w != contextWord)
              w1 = Math.min(w, contextWord)
              w2 = Math.max(w, contextWord)
            } {
              coocurences += (w1, w2) -> (coocurences.getOrElse((w1, w2), 0.0) + 1.0 / (i + 1))
            }
            windowBuffer ::= w
            if (windowBuffer.size == window) windowBuffer = windowBuffer.init
          })
        }
        coocurences.map { case (k, v) => (k._1, k._2, v) }.toSeq.iterator
      }
      }
    coocurrenceMatrix
      .flatMap(x=>x)
      .map(x=>((x._1, x._2), x._3))
      .reduceByKey(_ + _)
      .filter(x=>x._2>0)
      .map(x=>(x._1._1, x._1._2, x._2))
  }


  // 训练模型的代码
  private def doFit(dataset: RDD[Array[String]],
                    bcVocabHash: Broadcast[mutable.HashMap[String, Int]],
                    coMatrix:RDD[(Int, Int, Double)]
                   ): RDD[(String, String)] ={
    val sc: SparkContext = dataset.context
    // 初始化向量
    val initRandom = new XORShiftRandom(seed)
    val syn0Global: Array[Float] =Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize) // embedding向量初始化
    val syn1Global: Array[Float] = Array.fill[Float](vocabSize)(0.0f)  // 偏置项初始化
    // 测试代码
    if(vocabHash.contains("6277_201512")) {
      val startIndex: Int = vocabHash.getOrElse("6277_201512", 1)
      println(s" 初始化单词6277_201512的embedding向量为： ${syn0Global.slice(startIndex * vectorSize, (startIndex + 1) * vectorSize).map(x => x.toString).mkString(",")}")
    }//
    val alpha = learningRate
    for (k <- 1 to numIterations) {
      println(s"------迭代第${k}步的结果----------")
      val start_time =new Date().getTime
      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      // 分布式更新参数
      val partial: RDD[(Int, Array[Float], Float)] = coMatrix
        .mapPartitions { sentences: Iterator[(Int, Int, Double)] =>
          val syn0Modify: Array[Float] = bcSyn0Global.value
          val syn1Modify: Array[Float] = bcSyn1Global.value
          sentences.foreach { x: (Int, Int, Double) =>
            val l1 = x._1 * vectorSize //w1的起始位置
            val l2 = x._2 * vectorSize //w2的起始位置
            val count = x._3              //两个词共现的次数
            val prediction: Float = blas.sdot(vectorSize, syn0Modify, l1, 1, syn0Modify, l2, 1)
            //            val wordAnorm = math.sqrt(blas.sdot(vectorSize, syn0Modify, l1, 1, syn0Modify, l1, 1)) // 一旦模为0，就会出现NaN和infinit的情况。
            //            val wordBnorm = math.sqrt(blas.sdot(vectorSize, syn0Modify, l2, 1, syn0Modify, l2, 1))
            //            //  进行归一化
            //            val word1Normlize: Array[Float] = if(wordAnorm-0.0<=1e-7){
            //              Array.fill[Float](vectorSize)(0)
            //            }else{
            //              syn0Modify
            //                .slice(x._1 * vectorSize, (x._1+1) * vectorSize)
            //                .map(x=>x/wordAnorm.toFloat)
            //            }
            //            val word2Normlize: Array[Float] = if(wordBnorm-0.0<=1e-7){
            //              Array.fill[Float](vectorSize)(0)
            //            }else{
            //              syn0Modify
            //                .slice(x._2 * vectorSize, (x._2+1) * vectorSize)
            //                .map(x=>x/wordAnorm.toFloat)
            //            }
            // f函数
            val entryWeight = Math.pow(Math.min(1.0f, (count / maxCount)), alpha)
            // 实际不为LOSS，而是LOSS的导数
            val loss = entryWeight * (prediction+syn1Modify(x._1)+syn1Modify(x._2) - Math.log(count))
            //            println(s"---curLoss=${loss}, entryWeight=${entryWeight}, count=${count}, syn1Modify(x._1)=${syn1Modify(x._1)}, syn1Modify(x._2)=${syn1Modify(x._2)}, prediction=${prediction} -----")
            // 梯度下降进行求解
            for (i <- 0 until vectorSize) {
              //              syn0Modify(l1+i) = (word1Normlize(i)- learningRate * loss * word2Normlize(i)).toFloat
              //              syn0Modify(l2+i) = (word2Normlize(i)- learningRate * loss * word1Normlize(i)).toFloat
              syn0Modify(l1+i) = (syn0Modify(l1+i)- learningRate * loss * syn0Modify(l2+i)).toFloat
              syn0Modify(l2+i) = (syn0Modify(l2+i)- learningRate * loss * syn0Modify(l1+i)).toFloat
            }
            syn1Modify(x._1) -= (learningRate * loss).toFloat
            syn1Modify(x._2) -= (learningRate * loss).toFloat
          }
          Iterator.tabulate(vocabSize) { index =>
            Some(
              (
                index, // 索引
                syn0Modify.slice(index * vectorSize, (index + 1) * vectorSize), // 权重向量
                syn1Modify(index) // 偏置项
              )
            )
          }.flatten
        }
      // 将不同partition的向量进行合并更新
      // 权重向量的更新，实际上是对所有ID一样的向量相加，再除以次数
      val synAgg1 = partial.mapPartitions { iter =>
        iter.map { case (id, vec: Array[Float],bais) =>
          (id, (vec, 1))
        }
      }.reduceByKey { case ((v1: Array[Float], count1: Int), (v2: Array[Float], count2: Int)) =>
        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
        (v1, count1 + count2)
      }.map { case (id, (vec, count)) =>
        blas.sscal(vectorSize, 1.0f / count, vec, 1)
        (id, vec)
      }.collect()
      println(s"------迭代第${k}步耗时: ${new Date().getTime-start_time}----------")
      // 偏置项的更新
      val synAgg2: Array[(Int, Float)] = partial.mapPartitions { iter =>
        iter.map { case (id, vec: Array[Float],bais: Float) =>
          (id, (bais, 1))
        }
      }.reduceByKey { case ((v1: Float, count1: Int), (v2: Float, count2: Int)) =>
        val v = v1+v2
        (v, count1 + count2)
      }.map { case (id, (vec: Float, count)) =>
        (id, vec/count)
      }.collect()
      println("-----参数向量更新完的结果----------------")
      synAgg1.take(5).foreach(x=>println(s"${x._1}  :  ${x._2.map(x=>x.toString).mkString(",")}"))
      println("-----偏置项更新完的结果----------------")
      synAgg2.take(5).foreach(x=>println(s"${x._1}  :  ${x._2}"))
      for(i<- 0 until vocabSize){
        for(j<- 0 until vectorSize){
          syn0Global(i*vectorSize+j)=synAgg1(i)._2(j)
        }
        //        Array.copy(synAgg1(i)._2, 0, syn0Global,  vectorSize, vectorSize)
        syn1Global(i)=synAgg2(i)._2
      }
      bcSyn0Global.destroy(false)
      bcSyn1Global.destroy(false)
      //
      if(vocabHash.contains("6277_201512")) {
        val startIndex: Int = vocabHash.getOrElse("6277_201512", 1)
        println(s" ${k}步后，单词6277_201512的embedding向量为： ${syn0Global.slice(startIndex * vectorSize, (startIndex + 1) * vectorSize).map(x => x.toString).mkString(",")}")
      }
      //
    }

    val bcSyn0Global = sc.broadcast(syn0Global)

    val value = sc
      .parallelize(vocabHash.map(x=>(x._1, x._2)).toArray[(String,Int)])
      .map{x=>
        val word = x._1
        val wordIndex: Int = x._2
        val wordEmbed: Array[Float] = bcSyn0Global.value.slice(wordIndex*vectorSize,(wordIndex+1)*vectorSize)
        val wordEmbedNorm: Double = Math.sqrt(wordEmbed.map(x=>x*x).sum)
        (x._1, wordEmbed.map(x=>x/wordEmbedNorm).map(x=>x.toString).mkString(","))
      }
    value
  }

  // 按照指定的长度，将一维数组切分为多维数组
  def split[A](xs: List[A], n: Int): List[List[A]] = {
    if (xs.size <= n) xs :: Nil
    else (xs take n) :: split(xs drop n, n)
  }

}