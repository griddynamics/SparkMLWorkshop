import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.functions.{avg, col, count, length, lit, trim, when}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

import scala.math.Fractional.Implicits.infixFractionalOps

object DecisionTreeClassifier {
  def main(args: Array[String]) = {
    val spark = SparkSession
      .builder
      .appName("DecisionTreeClassifier")
      .config("spark.master", "local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val hadoopConfig: Configuration = spark.sparkContext.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)

    //Читаю все файлы
    var train_df = spark.read.format("csv")
      .option("header", "True")
      .option("sep", ",")
      .option("inferSchema", "true")
      .load("src/main/data/train.csv")
    var test_df = spark.read.format("csv")
      .option("header", "True")
      .option("sep", ",")
      .option("inferSchema", "true")
      .load("src/main/data/test.csv")
    var test_result_df = spark.read.format("csv")
      .option("header", "True")
      .option("sep", ",")
      .option("inferSchema", "true")
      .load("src/main/data/is_survived.csv")

    //Вычисляю среднее значение возраста для заполнение пустых ячеек в этом столбце
    val mean_Age = train_df.select(avg(train_df("Age")))
    mean_Age.cache()
    val mean_Age_test = test_df.select(avg(test_df("Age")))
    mean_Age_test.cache()

    //Проверяю отсутствующие значения в столбцах фичей
    train_df.columns.toSeq.foreach{x => println(train_df.filter(col(x).isNull).count(), x)}

    //Подсчитываю количество уникальных значений в столбце Embarked
    train_df.groupBy("Embarked").agg(count(col("PassengerId"))).show()

    //Заменяю отсутствующие значения в столбце Embarked на "S", так как это значение чаще всего встрачается
    train_df = train_df.withColumn("Embarked", when(col("Embarked").isNull, "S")
      .otherwise(col("Embarked")))
    test_df = test_df.withColumn("Embarked", when(col("Embarked").isNull, "S")
      .otherwise(col("Embarked")))

    //Замена отсутствующих возрастов средними
    train_df = train_df.withColumn("Age", when(col("Age").isNull,
      mean_Age.select(col("avg(Age)")).first.getDouble(0).toInt)
      .otherwise(col("Age")))
    test_df = test_df.withColumn("Age", when(col("Age").isNull,
      mean_Age_test.select(col("avg(Age)")).first.getDouble(0).toInt)
      .otherwise(col("Age")))

    //Поиск длины имен пассажиров (без пробелов)
    train_df = train_df.withColumn("Name", length(trim(col("Name"))))
    test_df = test_df.withColumn("Name", length(trim(col("Name"))))

    //Преобразование колоноки Embarked в числа
    val indexer = new StringIndexer()
      .setInputCols(Array("Pclass", "Name", "Sex", "Age", "Parch", "Embarked"))
      .setOutputCols(Array("ind_Pclass", "ind_Name", "ind_Sex", "ind_Age", "ind_Parch", "ind_Embarked"))
    val indexed_train_df = indexer.fit(train_df).transform(train_df)
    val indexed_test_df = indexer.fit(test_df).transform(test_df)


    //Создание общего вектора для всех фичей
    val assembler = new VectorAssembler()
      .setInputCols(Array("ind_Pclass", "ind_Name", "ind_Sex", "ind_Age", "ind_Parch", "ind_Embarked"))
      .setOutputCol("features")
    val featured_df = assembler.transform(indexed_train_df)
    val featured_df_test = assembler.transform(indexed_test_df)


    //Подсчитываем количество униклаьных значений во всех столбцах
    featured_df.columns.toSeq.foreach{x => println(featured_df.select(x).distinct().count(), x)}
    featured_df_test.columns.toSeq.foreach{x => println(featured_df_test.select(x).distinct().count(), x)}

    //Создание решающего дерева
    val dt = new DecisionTreeClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setMaxBins(100)
      .setMaxDepth(5)

    //Создание леса решающих деревьев
    val rf = new RandomForestClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setNumTrees(10)
      .setMaxBins(100)
      .setMaxDepth(5)
      .setPredictionCol("second_prediction")
      .setRawPredictionCol("second_raw_prediction")
      .setProbabilityCol("second_probability")

    //Создание пайплайна
    val pipeline = new Pipeline()
      .setStages(Array(dt, rf))

    //Обучение моделей
    val model = pipeline.fit(featured_df)

    //Предсказание на тестовых данных
    var prediction = model.transform(featured_df_test)

    //Добавление колонки Survived для проверки полученного результат
    prediction = prediction.join(test_result_df, Seq("PassengerId"), "left")

    prediction.select( "PassengerId", "features", "prediction", "second_prediction", "Survived").show(5)

    //Подсчет accuracy и precision и вовод их на экран
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val evaluator_precision = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    val evaluator_2 = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("second_prediction")
      .setMetricName("accuracy")

    val evaluator_precision_2 = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("second_prediction")
      .setMetricName("weightedPrecision")

    val accuracy = evaluator.evaluate(prediction)
    val recall = evaluator_precision.evaluate(prediction)
    val accuracy_2 = evaluator_2.evaluate(prediction)
    val recall_2 = evaluator_precision_2.evaluate(prediction)

    println(s"accuracy of DecisionTree = ${(1.0 - accuracy)}")
    println(s"recall of DecisionTree = ${(1.0 - recall)}")
    println(s"accuracy of RandomForest = ${(1.0 - accuracy_2)}")
    println(s"recall of RandomForest = ${(1.0 - recall_2)}")

  }
}

