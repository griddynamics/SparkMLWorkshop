import org.apache.hadoop.conf.Configuration
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

object Test {
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

    val model = PipelineModel.load(modelPath)

    val res = model.transform(df)
    res.show(10, false)

  }
}
