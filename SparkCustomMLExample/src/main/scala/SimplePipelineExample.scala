import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression

object SimplePipelineExample {
  def main(args: Array[String]) = {
    val spark = SparkSession
      .builder
      .appName("SimplePipelineExample")
      .config("spark.master", "local")
      .getOrCreate()

    val hadoopConfig: Configuration = spark.sparkContext.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)

    import spark.implicits._

    val df = Seq(
      (0L, "money bingo", 1.0),
      (1L, "Good afternoon we are waiting for your answer", 0.0),
      (2L, "money money", 1.0)
    ).toDF("id", "in", "label")
    df.show()
    val tokenizer = new Tokenizer().setInputCol("in").setOutputCol("out")
    // HashingTF, maps a sequence of terms to their term frequencies using the hashing trick
    val hashingTF = new HashingTF().setNumFeatures(32).setInputCol(tokenizer.getOutputCol).setOutputCol("features")

    val lr = new LogisticRegression().setMaxIter(100).setRegParam(0.01)

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
//    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF))

  // Fit model
    val model = pipeline.fit(df)

    val res = model.transform(df)

    res.show(10, false)

//    VectorIndexer example
//    import org.apache.spark.ml.feature.VectorIndexer
//    import org.apache.spark.ml.linalg.Vectors
//    val data=Seq(Vectors.dense(-1,1,1,8,56),
//      Vectors.dense(-1,3,-1,-9,88),
//      Vectors.dense(0,5,1,10,96),
//      Vectors.dense(0,5,1,11,589))
//    val df2=spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
//    val indexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed").setMaxCategories(4) //.setMaxCategories(2)
//    val indexerModel = indexer.fit(df2)
//    val res2 = indexerModel.transform(df2)
//    res2.show(false)
//
//    val categoricalFeatures = indexerModel.categoryMaps
//    println(categoricalFeatures)
//    println(categoricalFeatures.size)
//
//    res2.printSchema
  }
}
