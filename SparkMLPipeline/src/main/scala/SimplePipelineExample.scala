import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
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
      (1L, "Good afternoon, we are waiting for your answer.", 0.0),
      (2L, "money money", 1.0)
    ).toDF("id", "in", "label")
    df.show()
    val tokenizer = new Tokenizer().setInputCol("in").setOutputCol("out")
    // HashingTF, maps a sequence of terms to their term frequencies using the hashing trick
    val hashingTF = new HashingTF().setNumFeatures(10).setInputCol(tokenizer.getOutputCol).setOutputCol("features")

    val lr = new LogisticRegression().setMaxIter(100).setRegParam(0.01)

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

    // Fit model
    val model = pipeline.fit(df)

    val modelPath = "model"
    // Save model
    model.write.overwrite().save(modelPath)

    // Read model
    val preparedModel = PipelineModel.load(modelPath)

    // Predict
    val res = preparedModel.transform(df)

    res.show(10, false)

  }
}
