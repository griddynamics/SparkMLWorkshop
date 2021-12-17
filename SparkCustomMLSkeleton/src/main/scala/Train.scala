import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{CustomEstimator, CustomTransformer}
import org.apache.spark.ml.Pipeline

object Train {
  def main(args: Array[String]) = {
    val spark = SparkSession
      .builder
      .appName("CustomPipelineExample")
      .config("spark.master", "local")
      .getOrCreate()

    val hadoopConfig: Configuration = spark.sparkContext.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)

    val modelPath = "customModel"

    import spark.implicits._

    // Read your df here
    // val df = ???

    val customTransformer = new CustomTransformer()

    val customEstimator = new CustomEstimator()

    val pipeline = new Pipeline().setStages(Array(customTransformer, customEstimator))

    val model = pipeline.fit(df)

    model.write.overwrite().save(modelPath)
  }
}
