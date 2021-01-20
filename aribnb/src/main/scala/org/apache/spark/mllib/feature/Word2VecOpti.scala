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

import java.lang.{Iterable => JavaIterable}
import scala.collection.JavaConverters._
import scala.collection.{immutable, mutable}
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.BoundedPriorityQueue
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

import scala.collection.mutable.ArrayBuffer



/**
 * Word2VecOpti creates vector representation of words in a text corpus.
 * The algorithm first constructs a vocabulary from the corpus
 * and then learns vector representation of words in the vocabulary.
 * The vector representation can be used as features in
 * natural language processing and machine learning algorithms.
 *
 * We used skip-gram model in our implementation and hierarchical softmax
 * method to train the model. The variable names in the implementation
 * matches the original C implementation.
 *
 * For original C implementation, see https://code.google.com/p/Word2VecOpti/
 * For research papers, see
 * Efficient Estimation of Word Representations in Vector Space
 * and
 * Distributed Representations of Words and Phrases and their Compositionality.
 */
@Since("1.1.0")
class Word2VecOpti extends Serializable with Logging {


  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 1
  private var numIterations = 1
  private var seed = Utils.random.nextLong()
  private var minCount = 5
  private var maxSentenceLength = 1000
  private var EOC=""

  def setEoc(EOC: String): this.type = {
    // 可以允许输入的初始化的参数是空的，此时就选择默认的初始化的方式
    println(EOC.isEmpty)
    this.EOC = EOC
    this
  }

  /**
   * Sets the maximum length (in words) of each sentence in the input data.
   * Any sentence longer than this threshold will be divided into chunks of
   * up to `maxSentenceLength` size (default: 1000)
   */
  @Since("2.0.0")
  def setMaxSentenceLength(maxSentenceLength: Int): this.type = {
    require(maxSentenceLength > 0,
      s"Maximum length of sentences must be positive but got ${maxSentenceLength}")
    this.maxSentenceLength = maxSentenceLength
    this
  }

  /**
   * Sets vector size (default: 100).
   */
  @Since("1.1.0")
  def setVectorSize(vectorSize: Int): this.type = {
    require(vectorSize > 0,
      s"vector size must be positive but got ${vectorSize}")
    this.vectorSize = vectorSize
    this
  }

  /**
   * Sets initial learning rate (default: 0.025).
   */
  @Since("1.1.0")
  def setLearningRate(learningRate: Double): this.type = {
    require(learningRate > 0,
      s"Initial learning rate must be positive but got ${learningRate}")
    this.learningRate = learningRate
    this
  }

  /**
   * Sets number of partitions (default: 1). Use a small number for accuracy.
   */
  @Since("1.1.0")
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0,
      s"Number of partitions must be positive but got ${numPartitions}")
    this.numPartitions = numPartitions
    this
  }

  /**
   * Sets number of iterations (default: 1), which should be smaller than or equal to number of
   * partitions.
   */
  @Since("1.1.0")
  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations >= 0,
      s"Number of iterations must be nonnegative but got ${numIterations}")
    this.numIterations = numIterations
    this
  }

  /**
   * Sets random seed (default: a random long integer).
   */
  @Since("1.1.0")
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * Sets the window of words (default: 5)
   */
  @Since("1.6.0")
  def setWindowSize(window: Int): this.type = {
    require(window > 0,
      s"Window of words must be positive but got ${window}")
    this.window = window
    this
  }

  /**
   * Sets minCount, the minimum number of times a token must appear to be included in the Word2VecOpti
   * model's vocabulary (default: 5).
   */
  @Since("1.3.0")
  def setMinCount(minCount: Int): this.type = {
    require(minCount >= 0,
      s"Minimum number of times must be nonnegative but got ${minCount}")
    this.minCount = minCount
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_CODE_LENGTH = 40

  /** context words from [-window, window] */
  private var window = 5

  private var trainWordsCount = 0L //训练词的个数，不去重统计，比如单词A出现10次，则总计词频的时候就未10，和vocabSize的区别是一个未去重统计一个去重统计
  private var vocabSize = 0 // 词的个数
  @transient private var vocab: Array[VocabWord] = null
  @transient private var vocabHash = mutable.HashMap.empty[String, Int]

  private def learnVocab[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    //TODO STEP1：将-1的ID 去除，因为-1代表的是尾部的商品是下单的商品
    val words = dataset.flatMap(x => x)
      .filter(x=>x!=EOC)//构造词典的时候不将-1加入，该字符当做终止字符
      .filter(x=>x!=" ")//乱码符号剔除
      .filter(x=>x.nonEmpty)//构造词典的时候不将-1加入，该字符当做终止字符

    //统计每个词的词频，并且按照从大到小的顺序排列
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .map(x => VocabWord(
        x._1,
        x._2,
        new Array[Int](MAX_CODE_LENGTH),
        new Array[Int](MAX_CODE_LENGTH),
        0))
      .collect()
      .sortBy(_.cn)(Ordering[Long].reverse)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    // 构造HASH表，也就是单词和其下标的对应关系。比如单词A的词频最高，则其对应的索引为0，依次类推，按词频从高到低构建索引
    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }
    logInfo(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }

  // 构造HUFFMAN树
  private def createBinaryTree(): Unit = {
    val count = new Array[Long](vocabSize * 2 + 1)
    val binary = new Array[Int](vocabSize * 2 + 1)
    val parentNode = new Array[Int](vocabSize * 2 + 1)
    val code = new Array[Int](MAX_CODE_LENGTH)
    val point = new Array[Int](MAX_CODE_LENGTH)
    var a = 0
    while (a < vocabSize) {
      count(a) = vocab(a).cn
      a += 1
    }
    while (a < 2 * vocabSize) {
      count(a) = Long.MaxValue
      a += 1
    }
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0

    a = 0
    while (a < vocabSize - 1) {
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      assert(count(min1i) < Long.MaxValue)
      assert(count(min2i) < Long.MaxValue)
      count(vocabSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      binary(min2i) = 1
      a += 1
    }
    // Now assign binary code to each vocabulary word
    var i = 0
    a = 0
    while (a < vocabSize) {
      var b = a
      i = 0
      while (b != vocabSize * 2 - 2) {
        code(i) = binary(b)
        point(i) = b
        i += 1
        b = parentNode(b)
      }
      vocab(a).codeLen = i
      vocab(a).point(0) = vocabSize - 2
      b = 0
      while (b < i) {
        vocab(a).code(i - b - 1) = code(b)
        vocab(a).point(i - b) = point(b) - vocabSize
        b += 1
      }
      a += 1
    }
  }

  /**
   * Computes the vector representation of each word in vocabulary.
   * @param dataset an RDD of sentences,
   *                each sentence is expressed as an iterable collection of words
   * @return a Word2VecOptiModel
   */
  @Since("1.1.0")
  def fit[S <: Iterable[String]](dataset: RDD[S]): Word2VecOptiModel = {


    learnVocab(dataset)

    createBinaryTree()

    val sc = dataset.context

    val expTable = sc.broadcast(createExpTable())
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)

    try {
      doFit(dataset, sc, expTable, bcVocab, bcVocabHash)
    } finally {
      expTable.destroy(blocking = false)
      bcVocab.destroy(blocking = false)
      //      bcVocabHash.destroy(blocking = false)//已经释放了就不要再释放了
    }
  }

  private def doFit[S <: Iterable[String]](
                                            dataset: RDD[S], sc: SparkContext,
                                            expTable: Broadcast[Array[Float]],
                                            bcVocab: Broadcast[Array[VocabWord]],
                                            bcVocabHash: Broadcast[mutable.HashMap[String, Int]]) = {
    println(s"----总词的个数----- $vocabSize")
    // 初始化输入参数和模型参数
    // 原始代码如下：
    val initRandom = new XORShiftRandom(seed)
    val syn0Global =Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    val syn1Global = new Array[Float](vocabSize * vectorSize)
    // each partition is a collection of sentences,
    // will be translated into arrays of Index integer
    // 将词RDD变成其索引构成的RDD
    val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
      // Each sentence will map to 0 or more Array[Int]
      sentenceIter.flatMap { sentence =>
        //此处代码的目的是为了将单词对应到索引上，变成索引矩阵，其中不在词典中的，但是尾部词为-1的要进行保留，并给索引-1，后续为了方便表示
        // Sentence of words, some of which map to a word index
        //val wordIndexes: Iterable[Int] = sentence.flatMap(bcVocabHash.value.get)
        val bcHash = bcVocabHash.value
        val wordIndexes = new ArrayBuffer[Int]()
        for(curStr<-sentence){
          if(bcHash.contains(curStr)){
            wordIndexes.append(bcHash.getOrElse(curStr,-3))
          }
          else if(curStr==EOC){
            wordIndexes.append(-1)
          }
        }
        // break wordIndexes into trunks of maxSentenceLength when has more
        wordIndexes.grouped(maxSentenceLength).map(_.toArray)
      }
    }

    val newSentences: RDD[Array[Int]] = sentences.repartition(numPartitions).cache()

    // 总词*词的DIM若比最大的整数还大，说明不支持计算了，建议调节mincount的值，丢弃部分低频词
    if (vocabSize.toLong * vectorSize >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2VecOpti" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue`.")
    }


    // 总词个数，等于迭代次数*trainWordsCount+1  不明白为什么要+1 ？？？？
    val totalWordsCounts = numIterations * trainWordsCount + 1
    var alpha = learningRate

    for (k <- 1 to numIterations) {
      println(s"------迭代第${k}步的结果----------")
      val bcSyn0Global = sc.broadcast(syn0Global)//对应 𝑥𝑤，是输入向量
      val bcSyn1Global = sc.broadcast(syn1Global)//对应 𝜃𝑖𝑗−1 ，是参数向量
      val numWordsProcessedInPreviousIterations = (k - 1) * trainWordsCount

      val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
      val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
      val syn0Modify = new Array[Int](vocabSize)
      val syn1Modify = new Array[Int](vocabSize)
      val model: (Array[Float], Array[Float], Long, Long) = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, 0L, 0L)) {
        case ((syn0, syn1, lastWordCount, wordCount), sentence) =>
          var lwc = lastWordCount
          var wc = wordCount
          if (wordCount - lastWordCount > 10000) {
            lwc = wordCount
            //学习率变化的过程
            alpha = learningRate *
              (1 - (numPartitions * wordCount.toDouble + numWordsProcessedInPreviousIterations) /
                totalWordsCounts)
            if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
            logInfo(s"wordCount = ${wordCount + numWordsProcessedInPreviousIterations}, " +
              s"alpha = $alpha")
          }
          // 若最终的词为EOC，则将当前词的长度减1
          var curLength = sentence.length
          val lastIndex: Int = sentence(curLength-1)
          if(lastIndex == -1){
            curLength -=1
            println("\n\n===========尾部是-1，表示倒数第二个词要作为窗口词进行更新===========")
          }
          else {
            println("\n\n===================尾部不是-1===========")
          }
          wc += curLength
          // 最后一个词-1不会作为中心词进行更新
        for(pos<- 0 until curLength if !(lastIndex == -1 && pos == curLength - 1)) {
          //此word是中心词
          val word: Int = sentence(pos)
          // 窗口词更新的过程中并不是窗口内所有词都更新，只随机更新设定窗口大小内的随机的一个窗口。比如window=4, b=2,则只更新-2，-1，1，2。b=3，则只更新-1，1
          val b = random.nextInt(window)
          print(s"\n----中心词为:${bcVocab.value(word).word},随机数为：b=$b 则$pos 的窗口大小为${window-b}的词下标为：")
          // Train Skip-gram
          // TODO 定义一个遍历的窗口词的数组，若尾部词是-1，说明倒数第二个词是下单词，则倒数第二个词要被当做窗口词进行更新
          val tmp1 = (b until window * 2 + 1 - b).toArray
          val addInt = curLength-1-pos+window
          val contenxtIndexArr: Seq[Int] = if(lastIndex == -1 && !tmp1.contains(addInt)){
            tmp1:+addInt
          }else{
            tmp1
          }
          // 参数更新过程见https://www.cnblogs.com/pinard/p/7243513.html
          // 实际上是以b为中心，更新其左右窗口大小为window-b的所有词。例如窗口大小等于5，b=2，实际上只更新中心词上下文4个词
          for(a <- contenxtIndexArr){
            if (a != window) {
              val c = pos - window + a
              //说明上下文中的词，还在sentence内部
              if (c >= 0 && c < curLength) {
                print(s"$c ,")
                val lastWord: Int = sentence(c) //也就是上下文词的索引
                val l1 = lastWord * vectorSize // 在参数向量的起始位置
                val neu1e = new Array[Float](vectorSize) // 对应e向量。
                // Hierarchical softmax
                var d = 0
                // 更新HUFFMAN内部节点的参数
                while (d < bcVocab.value(word).codeLen) {
                  val inner: Int = bcVocab.value(word).point(d) //路径上的节点index
                  // 输入向量的起始位置
                  val l2 = inner * vectorSize
                  // Propagate hidden -> output
                  //向量点乘，syn0 .* syn1
                  var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                  if (f > -MAX_EXP && f < MAX_EXP) {
                    val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                    f = expTable.value(ind)
                    val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                    //neu1e = g * syn1 + neu1e, 倒数第二个参数为0的原因是e向量作为临时向量进行计算，其大小就只有vectorSize
                    blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                    //syn1 = g * syn0 + syn1
                    blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                    syn1Modify(inner) += 1
                  }
                  d += 1
                  // syn0 = syn0+neu1e
                  blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                  syn0Modify(lastWord) += 1 //保存syn0的迭代变化次数
                }
              }
            }
          }
        }
          (syn0, syn1, lwc, wc)
        }
        val syn0Local = model._1
        val syn1Local = model._2
        // Only output modified vectors.
        Iterator.tabulate(vocabSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index, syn0Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1Modify(index) > 0) {
            Some((index + vocabSize, syn1Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten
      }
      // SPARK-24666: do normalization for aggregating weights from partitions.
      // Original Word2VecOpti either single-thread or multi-thread which do Hogwild-style aggregation.
      // Our approach needs to do extra normalization, otherwise adding weights continuously may
      // cause overflow on float and lead to infinity/-infinity weights.
      val synAgg = partial.mapPartitions { iter =>
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
      var i = 0
      while (i < synAgg.length) {
        val index = synAgg(i)._1
        if (index < vocabSize) {
          Array.copy(synAgg(i)._2, 0, syn0Global, index * vectorSize, vectorSize)
        } else {
          Array.copy(synAgg(i)._2, 0, syn1Global, (index - vocabSize) * vectorSize, vectorSize)
        }
        i += 1
      }
      bcSyn0Global.destroy(false)
      bcSyn1Global.destroy(false)
    }
    newSentences.unpersist()

    val wordArray = vocab.map(_.word)
    new Word2VecOptiModel(wordArray.zipWithIndex.toMap, syn0Global)
  }

  /**
   * Computes the vector representation of each word in vocabulary (Java version).
   * @param dataset a JavaRDD of words
   * @return a Word2VecOptiModel
   */
  @Since("1.1.0")
  def fit[S <: JavaIterable[String]](dataset: JavaRDD[S]): Word2VecOptiModel = {
    fit(dataset.rdd.map(_.asScala))
  }
}

/**
 * Word2VecOpti model
 * @param wordIndex maps each word to an index, which can retrieve the corresponding
 *                  vector from wordVectors
 * @param wordVectors array of length numWords * vectorSize, vector corresponding
 *                    to the word mapped with index i can be retrieved by the slice
 *                    (i * vectorSize, i * vectorSize + vectorSize)
 */
@Since("1.1.0")
class Word2VecOptiModel private[spark] (
                                         private[spark] val wordIndex: Map[String, Int],
                                         private[spark] val wordVectors: Array[Float]) extends Serializable with Saveable {

  private val numWords = wordIndex.size
  // vectorSize: Dimension of each word's vector.
  private val vectorSize = wordVectors.length / numWords

  // wordList: Ordered list of words obtained from wordIndex.
  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }

  // wordVecNorms: Array of length numWords, each value being the Euclidean norm
  //               of the wordVector.
  private val wordVecNorms: Array[Float] = {
    val wordVecNorms = new Array[Float](numWords)
    var i = 0
    while (i < numWords) {
      val vec = wordVectors.slice(i * vectorSize, i * vectorSize + vectorSize)
      wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)
      i += 1
    }
    wordVecNorms
  }

  @Since("1.5.0")
  def this(model: Map[String, Array[Float]]) = {
    this(Word2VecOptiModel.buildWordIndex(model), Word2VecOptiModel.buildWordVectors(model))
  }

  override protected def formatVersion = "1.0"

  @Since("1.4.0")
  def save(sc: SparkContext, path: String): Unit = {
    Word2VecOptiModel.SaveLoadV1_0.save(sc, path, getVectors)
  }

  /**
   * Transforms a word to its vector representation
   * @param word a word
   * @return vector representation of word
   */
  @Since("1.1.0")
  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors.slice(ind * vectorSize, ind * vectorSize + vectorSize)
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  /**
   * Find synonyms of a word; do not include the word itself in results.
   * @param word a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
  @Since("1.1.0")
  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num, Some(word))
  }

  /**
   * Find synonyms of the vector representation of a word, possibly
   * including any words in the model vocabulary whose vector respresentation
   * is the supplied vector.
   * @param vector vector representation of a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
  @Since("1.1.0")
  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    findSynonyms(vector, num, None)
  }

  /**
   * Find synonyms of the vector representation of a word, rejecting
   * words identical to the value of wordOpt, if one is supplied.
   * @param vector vector representation of a word
   * @param num number of synonyms to find
   * @param wordOpt optionally, a word to reject from the results list
   * @return array of (word, cosineSimilarity)
   */
  private def findSynonyms(
                            vector: Vector,
                            num: Int,
                            wordOpt: Option[String]): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")

    val fVector = vector.toArray.map(_.toFloat)
    val cosineVec = new Array[Float](numWords)
    val alpha: Float = 1
    val beta: Float = 0
    // Normalize input vector before blas.sgemv to avoid Inf value
    val vecNorm = blas.snrm2(vectorSize, fVector, 1)
    if (vecNorm != 0.0f) {
      blas.sscal(vectorSize, 1 / vecNorm, fVector, 0, 1)
    }
    blas.sgemv(
      "T", vectorSize, numWords, alpha, wordVectors, vectorSize, fVector, 1, beta, cosineVec, 1)

    var i = 0
    while (i < numWords) {
      val norm = wordVecNorms(i)
      if (norm == 0.0f) {
        cosineVec(i) = 0.0f
      } else {
        cosineVec(i) /= norm
      }
      i += 1
    }

    val pq = new BoundedPriorityQueue[(String, Float)](num + 1)(Ordering.by(_._2))

    var j = 0
    while (j < numWords) {
      pq += Tuple2(wordList(j), cosineVec(j))
      j += 1
    }

    val scored = pq.toSeq.sortBy(-_._2)

    val filtered = wordOpt match {
      case Some(w) => scored.filter(tup => w != tup._1)
      case None => scored
    }

    filtered
      .take(num)
      .map { case (word, score) => (word, score.toDouble) }
      .toArray
  }

  /**
   * Returns a map of words to their vector representations.
   */
  @Since("1.2.0")
  def getVectors: Map[String, Array[Float]] = {
    wordIndex.map { case (word, ind) =>
      (word, wordVectors.slice(vectorSize * ind, vectorSize * ind + vectorSize))
    }
  }

}

@Since("1.4.0")
object Word2VecOptiModel extends Loader[Word2VecOptiModel] {

  private def buildWordIndex(model: Map[String, Array[Float]]): Map[String, Int] = {
    model.keys.zipWithIndex.toMap
  }

  private def buildWordVectors(model: Map[String, Array[Float]]): Array[Float] = {
    require(model.nonEmpty, "Word2VecOptiMap should be non-empty")
    val (vectorSize, numWords) = (model.head._2.length, model.size)
    val wordList = model.keys.toArray
    val wordVectors = new Array[Float](vectorSize * numWords)
    var i = 0
    while (i < numWords) {
      Array.copy(model(wordList(i)), 0, wordVectors, i * vectorSize, vectorSize)
      i += 1
    }
    wordVectors
  }

  private object SaveLoadV1_0 {

    val formatVersionV1_0 = "1.0"

    val classNameV1_0 = "org.apache.spark.mllib.feature.Word2VecOptiModel"

    case class Data(word: String, vector: Array[Float])

    def load(sc: SparkContext, path: String): Word2VecOptiModel = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val dataFrame = spark.read.parquet(Loader.dataPath(path))
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      Loader.checkSchema[Data](dataFrame.schema)

      val dataArray = dataFrame.select("word", "vector").collect()
      val Word2VecOptiMap = dataArray.map(i => (i.getString(0), i.getSeq[Float](1).toArray)).toMap
      new Word2VecOptiModel(Word2VecOptiMap)
    }

    def save(sc: SparkContext, path: String, model: Map[String, Array[Float]]): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()

      val vectorSize = model.values.head.length
      val numWords = model.size
      val metadata = compact(render(
        ("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~
          ("vectorSize" -> vectorSize) ~ ("numWords" -> numWords)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

      // We want to partition the model in partitions smaller than
      // spark.kryoserializer.buffer.max
      val bufferSize = Utils.byteStringAsBytes(
        spark.conf.get("spark.kryoserializer.buffer.max", "64m"))
      // We calculate the approximate size of the model
      // We only calculate the array size, considering an
      // average string size of 15 bytes, the formula is:
      // (floatSize * vectorSize + 15) * numWords
      val approxSize = (4L * vectorSize + 15) * numWords
      val nPartitions = ((approxSize / bufferSize) + 1).toInt
      val dataArray = model.toSeq.map { case (w, v) => Data(w, v) }
      spark.createDataFrame(dataArray).repartition(nPartitions).write.parquet(Loader.dataPath(path))
    }
  }

  @Since("1.4.0")
  override def load(sc: SparkContext, path: String): Word2VecOptiModel = {

    val (loadedClassName, loadedVersion, metadata) = Loader.loadMetadata(sc, path)
    implicit val formats = DefaultFormats
    val expectedVectorSize = (metadata \ "vectorSize").extract[Int]
    val expectedNumWords = (metadata \ "numWords").extract[Int]
    val classNameV1_0 = SaveLoadV1_0.classNameV1_0
    (loadedClassName, loadedVersion) match {
      case (classNameV1_0, "1.0") =>
        val model = SaveLoadV1_0.load(sc, path)
        val vectorSize = model.getVectors.values.head.length
        val numWords = model.getVectors.size
        require(expectedVectorSize == vectorSize,
          s"Word2VecOptiModel requires each word to be mapped to a vector of size " +
            s"$expectedVectorSize, got vector of size $vectorSize")
        require(expectedNumWords == numWords,
          s"Word2VecOptiModel requires $expectedNumWords words, but got $numWords")
        model
      case _ => throw new Exception(
        s"Word2VecOptiModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $loadedVersion).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }
  }
}
