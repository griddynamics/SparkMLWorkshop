import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.LocalFileSystem
import org.apache.hadoop.hdfs.DistributedFileSystem
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

object TitanicPipeline {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder
      .appName("SimplePipelineExample")
      .config("spark.master", "local")
      .config("spark.driver.cores", 2)
      .config("spark.executor.cores", 2)
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val hadoopConfig: Configuration = spark.sparkContext.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[LocalFileSystem].getName)

    val preparedTrainData: DataFrame = prepare {
      spark.read.option("header", "true").csv("data/titanic/train.csv")
    }
    println(s"Train data size: ${preparedTrainData.count}")

    val preparedTestData = prepare {
      spark.read.option("header", "true").csv("data/titanic/test.csv")
        .join(
          spark.read.option("header", "true").csv("data/titanic/is_survived.csv"),
          Seq("PassengerId")
        )
    }
    println(s"Test data size: ${preparedTestData.count}")

    // Split the data into training and control sets (30% held out for testing)
    val Array(trainingData, controlData) = preparedTrainData.randomSplit(Array(0.65, 0.35), seed = 123L)

    // The stages of our pipeline
    val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex")
    val embarkedIndexer = new StringIndexer().setInputCol("embarked").setOutputCol("embarkedIndex")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("age", "siblings", "family", "fare", "cabin", "title", "sexIndex", "class", "embarkedIndex"))
      .setOutputCol("features")

    //val normalizer = new Normalizer().setInputCol(vectorAssembler.getOutputCol).setOutputCol("features").setP(2.0)

    val classifications = Seq(
      new LogisticRegression()
        .setRegParam(0.1)
        .setMaxIter(9)
        .setFamily("multinomial"),
      new NaiveBayes().setModelType("gaussian"),
      new RandomForestClassifier().setNumTrees(27),
      new DecisionTreeClassifier(),
      new GBTClassifier()
        .setMaxIter(18)
        .setFeatureSubsetStrategy("sqrt"),
      new MultilayerPerceptronClassifier()
        .setLayers(Array[Int](9, 5, 9, 2))
        .setBlockSize(128)
        .setSeed(1234L)
        .setMaxIter(100)
    )

    val results = classifications.flatMap {
      classifier =>
        val pipeline = new Pipeline().setStages(
          Array(sexIndexer, embarkedIndexer, vectorAssembler, classifier)
        )
        execute(trainData = trainingData, controlData = controlData)(testData = preparedTestData)(classifier, pipeline)
    }
    preparedTrainData.unpersist()
    preparedTestData.unpersist()

    results.sortBy(-_.accuracy).foreach(r => println(r))

    println(s"Best result: ${results.maxBy(_.accuracy)}")

  }

  def execute(trainData: Dataset[Row], controlData: Dataset[Row])(testData: DataFrame)(
    classification: PipelineStage, pipeline: Pipeline)(implicit spark: SparkSession)
  : Seq[PredictResult] = {


    val className = classification.getClass.getSimpleName
    println(s"Execute: $className")

    val modelPath = s"model/$className"

    // Fit model
    val model = pipeline.fit(trainData)
    // Save model
    model.write.overwrite().save(modelPath)

    Seq(
      predict(className, controlData, "train", modelPath),
      predict(className, testData, "test", modelPath)
    )
  }

  case class PredictResult(className: String, setName: String, eq: Long, all: Long, accuracy: Double) {
    override def toString: String = {
      s"$eq/$all  Accuracy = $accuracy, Error = ${1.0 - accuracy} $className $setName"
    }
  }

  def prepare(df: DataFrame)(implicit spark: SparkSession): DataFrame = {

    import spark.implicits._

    val to1WhenNA = udf((a: Any) => if (Option(a).contains("NA")) 1 else 0)

    val naDF = (if (df.columns.contains("Survived")) {
      df
        .withColumn("label", $"Survived".cast(DoubleType))
        .drop("Survived")
    } else
      df
      ).na.fill("NA")

    naDF.agg(
      sum(to1WhenNA($"Pclass")).as("NA Pclass"),
      sum(to1WhenNA($"Name")).as("NA Name"),
      sum(to1WhenNA($"Sex")).as("NA Sex"),
      sum(to1WhenNA($"Age")).as("NA Age"),
      sum(to1WhenNA($"SibSp")).as("NA SibSp"),
      sum(to1WhenNA($"Parch")).as("NA Parch"),
      sum(to1WhenNA($"Ticket")).as("NA Ticket"),
      sum(to1WhenNA($"Fare")).as("NA Fare"),
      sum(to1WhenNA($"Cabin")).as("NA Cabin"),
      sum(to1WhenNA($"Embarked")).as("NA Embarke")
    ).show()

    //    naDF.groupBy($"Age").count.show()
    //    naDF.groupBy($"Cabin").count.show()
    //    naDF.groupBy($"Embarked").count.show()

    val emDF = naDF.withColumn("Embarked", when($"Embarked" === "NA", "S").otherwise($"Embarked"))
      .withColumn("hasCabin", when($"Cabin" === "NA", 0).otherwise(1))

    val titles = List("Capt.", "Col.", "Countess.", "Don.", "Dona.", "Dr.", "Jonkheer.", "Lady.", "Major.", "Master.", "Miss.", "Mlle.", "Mme.", "Mr.", "Mrs.", "Ms.", "Rev.", "Sir.").zipWithIndex.toMap
    val titleToInt = udf((s: String) => Option(s).flatMap(titles.get).getOrElse(0))

    val titleDF = emDF.withColumn("title", titleToInt(regexp_extract($"Name", "\\w+\\.", 0)).cast(IntegerType))

    val meanAges = titleDF.groupBy($"title")
      .agg(
        count($"Age"),
        sum(when($"Age" === "NA", 1).otherwise(0)).as("count(Age=NA)"),
        round(mean($"Age"), 2).as("mean")
      )
    // meanAges.show(100, false)
    val ages = meanAges
      .select($"title", coalesce($"mean", lit(29.75)))
      .map(r => r.getInt(0) -> r.getDouble(1))
      .collect().toMap

    // println(ages)

    def setMeanFromInt(m: Map[Int, Double]) = udf((t: Int) => m.get(t))

    def setMeanFromString(m: Map[String, Double]) = udf((t: String) => m.get(t))

    val fullAgeDF = titleDF.withColumn("age", when($"Age" === "NA", setMeanFromInt(ages)($"title").cast(DoubleType)).otherwise($"Age".cast(DoubleType)))
    //    fullAgeDF.groupBy($"title")
    //      .agg(
    //        count($"age"),
    //        sum(when($"age" === "NA", 1).otherwise(0)).as("count(Age=NA)"),
    //        round(mean($"age"), 2).as("mean")
    //      )
    //      .show(100, false)

    val fares = fullAgeDF.where($"Fare".as[Double] > 0)
      .groupBy("Pclass").agg(round(mean($"Fare".as[Double]), 4).as("mean"))
      .select($"Pclass", coalesce($"mean", lit(33.0)))
      .map(r => r.getString(0) -> r.getDouble(1))
      .collect().toMap

    //println(fares)

    //fullAgeDF.where($"Fare".isNull || $"Pclass".isNull).show(false)

    val prepared = fullAgeDF
      .withColumn("fare", when($"Fare".isNull || $"Fare" === "NA" || $"Fare".as[Double] <= 0, setMeanFromString(fares)($"Pclass")).otherwise($"Fare".cast(DoubleType)))
      .withColumn("class", $"Pclass".cast(DoubleType))
      .withColumn("siblings", $"SibSp".cast(IntegerType))
      .withColumn("family", $"Parch".cast(IntegerType))
      .withColumn("cabin", $"hasCabin".cast(IntegerType))
      .withColumnRenamed("Sex", "sex")
      .withColumnRenamed("Embarked", "embarked")
      .drop("PassengerId", "Pclass", "Name", "SibSp", "Parch", "Ticket")
      .persist(StorageLevel.MEMORY_AND_DISK)

    // prepared.printSchema()
    prepared.describe(prepared.columns: _*).show(false)
    val check = prepared.where($"sex".isNull || $"age".isNull || $"fare".isNull || $"cabin".isNull || $"embarked".isNull || $"label".isNull || $"hasCabin".isNull || $"title".isNull || $"class".isNull || $"siblings".isNull || $"family".isNull)
    if (check.count() > 0) check.show(10, false)
    prepared
  }

  def predict(className: String, testData: Dataset[Row], setName: String, modelPath: String)(implicit spark: SparkSession): PredictResult = {
    import spark.implicits._

    // Read model
    val preparedModel = PipelineModel.load(modelPath)

    // Predict
    val predicted = preparedModel.transform(testData).cache()

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val res = PredictResult(
      className,
      setName,
      eq = predicted.where($"label" === $"prediction").count,
      all = predicted.count,
      accuracy = evaluator.evaluate(predicted)
    )

    predicted.unpersist()
    res
  }
}
