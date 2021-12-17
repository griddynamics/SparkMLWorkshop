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

    /**
     * id: unique user identifier
     * subscriptionType: 0 - no subscription, 1 - basic subscription, 2 - premium subscription
     * favoriteGenres: 0 - Action, 1 - Comedy, 2 - Drama, 3 - Fantasy, 4 - Horror, 5 - Mystery, 6 - Romance, 7 - Thriller, 8 - Western
     * promoCodes: number of used promotional codes
     * timePerMonth: average time in hours spent watching
     * numberOfRatings: number of grades
     * spendPerMonth: the amount of money spent per month
     */
    val df = Seq(
      (1754, 1, 2, 0, 6, 5, 100),
      (5647, 0, 8, 3, 8, 2, 0),
      (2645, 1, 6, 2, 12, 1, 247),
      (9685, 1, 5, 0, 12, 4, 1495),
      (4756, 2, 2, 0, 3, 3, 203),
      (2484, 2, 0, 2, 5, 0, 3841)
    ).toDF("id", "subscriptionType", "favoriteGenres", "promoCodes", "timePerMonth", "numberOfRatings", "spendPerMonth")

    val customTransformer = new CustomTransformer()

    val customEstimator = new CustomEstimator()

    val pipeline = new Pipeline().setStages(Array(customTransformer, customEstimator))

    val model = pipeline.fit(df)

    model.write.overwrite().save(modelPath)
  }
}
