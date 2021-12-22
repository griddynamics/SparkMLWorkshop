import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.fpm.FPGrowth

object FPGrowth {
  def main(args: Array[String]) = {
    val spark = SparkSession
      .builder
      .appName("FPGrowth")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

    val hadoopConfig: Configuration = spark.sparkContext.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)

    val dataset = spark.createDataset(Seq(
      "1 2 5",
      "1 2 3 5",
      "1 2")
    ).map(t => t.split(" ")).toDF("items")

    val fpgrowth = new FPGrowth().setItemsCol("items")
    .setMinSupport(0.5) //the minimum support for an itemset to be identified as frequent.
                        // For example, if an item appears 3 out of 5 transactions, it has a support of 3/5=0.6.
    .setMinConfidence(0.6) //minimum confidence for generating Association Rule.
                           // Confidence is an indication of how often an association rule has been found to be true.
                           // For example, if in the transactions itemset X appears 4 times,
                           // X and Y co-occur only 2 times, the confidence for the rule X => Y is then 2/4 = 0.5.
                           // The parameter will not affect the mining for frequent itemsets,
                           // but specify the minimum confidence for generating association rules from frequent 
    val model = fpgrowth.fit(dataset)

    // Display frequent itemsets.
    model.freqItemsets.show()

    // Display generated association rules.
    model.associationRules.show()

    // transform examines the input items against all the association rules and summarize the
    // consequents as prediction
    model.transform(dataset).show()

  }
}
