package org.apache.spark.mllib.feature

/**
 * Entry in vocabulary
 */
private case class VocabWord(
                              var word: String,
                              var cn: Long,
                              var point: Array[Int],
                              var code: Array[Int],
                              var codeLen: Int
                            )
