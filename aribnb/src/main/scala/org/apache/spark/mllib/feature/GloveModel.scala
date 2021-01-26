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
             learningRate:  Double = 0.05f,
             alpha:         Double = 0.75,
             maxCount:      Double = 100.0,
             seed:          Long   = 2L ,
             vectorSize:    Int    = 32,
             numIterations: Int    = 1
           ) extends Serializable with Logging {

  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var vocabSize: Int = 1
  private var vocab: Array[(String, Int)] = new Array[(String, Int)](vocabSize.toInt)


  // 计算序列的一些基础信息，词典的个数及每个词典中词的编号
  private def learnVocab(dataset: RDD[Array[String]]) = {
    val words: RDD[String] = dataset.flatMap(x => x)
    // 单词及其对应的频次，此处就涉及到内存的问题。此处的代码参考word2vec
    vocab= words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .collect()
      .sortBy(_._2)(Ordering[Int].reverse)
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
    val syn0Global: Array[Float] =Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    val syn1Global: Array[Float] = Array.fill[Float](vocabSize)((initRandom.nextFloat() - 0.5f) / vectorSize)

    val alpha = learningRate
    for (k <- 1 to numIterations) {
      println(s"------迭代第${k}步的结果----------")
      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val partial: RDD[(Int, Array[Float])] = coMatrix.mapPartitions { sentences: Iterator[(Int, Int, Double)] =>
        val syn0Modify: Array[Float] = bcSyn0Global.value
        val syn1Modify: Array[Float] = bcSyn1Global.value
        sentences.foreach { x: (Int, Int, Double) =>
          val l1 = x._1 * vectorSize //w1的起始位置
          val l2 = x._2 * vectorSize //w2的起始位置
          val count = x._3              //两个词共现的次数
          val prediction = blas.sdot(vectorSize, syn0Modify, l1, 1, syn0Modify, l2, 1)
          val word_a_norm = math.sqrt(blas.sdot(vectorSize, syn0Modify, l1, 1, syn0Modify, l1, 1))
          val word_b_norm = math.sqrt(blas.sdot(vectorSize, syn0Modify, l2, 1, syn0Modify, l2, 1))
          // f函数
          val entryWeight = Math.pow(Math.min(1.0, (count / maxCount)), alpha)
          // 实际不为LOSS，而是LOSS的导数
          val loss = entryWeight * (prediction - Math.log(count))
          // 梯度下降进行求解
          for (i <- 0 until vectorSize) {
            syn0Modify(l1+i) = (syn0Modify(l1+i)- learningRate * loss * syn0Modify(l2+i) / word_a_norm).toFloat
            syn0Modify(l2+i) = (syn0Modify(l2+i)- learningRate * loss * syn0Modify(l1+i) / word_b_norm).toFloat
          }
          syn1Modify(x._1) -= (learningRate * loss).toFloat
          syn1Modify(x._2) -= (learningRate * loss).toFloat
        }
        Iterator.tabulate(vocabSize) { index =>
            Some((index, syn0Modify.slice(index * vectorSize, (index + 1) * vectorSize)))
        }.flatten
      }
//      println(s"---总的生成向量数量为：${partial.count()}-----")
//      partial.take(500).foreach(x=>println(s"\n ${x._1.toString}的向量为: ${x._2.mkString(",")} \n"))

      //do normalization
      val synAgg: Array[(Int, Array[Float])] = partial.mapPartitions { iter =>
        iter.map { case (id, vec) =>
          (id, (vec, 1))
        }
      }.reduceByKey { case ((v1, count1), (v2, count2)) =>
        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
        (v1, count1 + count2)
      }.map { case (id, (vec, count)) =>
        blas.sscal(vectorSize, 1.0f / count, vec, 1)
        (id, vec)
      }.collect()

      for(i<- 0 until vocabSize){
        Array.copy(synAgg(i)._2, 0, syn0Global,  synAgg(i)._1* vectorSize, vectorSize)
      }
      bcSyn0Global.destroy(false)
      bcSyn1Global.destroy(false)
    }

    val bcSyn0Global = sc.broadcast(syn0Global)
    val value = sc
      .parallelize(vocab)
      .map{x=>
        val word: String = x._1
        val wordIndex: Int = x._2
        val wordEmbedArr: Array[Float] = bcSyn0Global.value
        val curWordEmbed: Array[Float] = new Array[Float](vectorSize)
        for(i<-0 until vectorSize){
          curWordEmbed(i)= wordEmbedArr(wordIndex*vectorSize+i)
        }
        (word, curWordEmbed.map(x=>x.toString).mkString(","))
      }
//
//
//
//    println("-------------将syn0Global按照vectorSize大小切分成二维数组---------------")
//    val tmpArr = split(syn0Global.toList, vectorSize)
//    val tuples = vocab.map(x => x._1).zip(tmpArr).map(x=>(x._1, x._2.map(x=>x.toString).mkString(",")))
//    val value = sc.parallelize(tuples)
    value
  }

  // 按照指定的长度，将一维数组切分为多维数组
  def split[A](xs: List[A], n: Int): List[List[A]] = {
    if (xs.size <= n) xs :: Nil
    else (xs take n) :: split(xs drop n, n)
  }

}