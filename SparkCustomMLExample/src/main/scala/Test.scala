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

    val df = Seq(
      (7564, 0, 0, 1, 3, 7),
      (2948, 0, 1, 2, 20, 4),
      (6758, 2, 5, 2, 3, 0),
      (1231, 2, 4, 3, 2, 2),
      (4955, 1, 3, 2, 3, 3),
      (3321, 2, 0, 1, 5, 0)
    ).toDF("id", "subscriptionType", "favoriteGenres", "promoCodes", "timePerMonth", "numberOfRatings")


    val model = PipelineModel.load(modelPath)

    val res = model.transform(df)
    res.show(10, false)

  }
}
