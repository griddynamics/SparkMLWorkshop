package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.feature.CustomEstimatorModel.CustomEstimatorModelWriter
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{col, _}
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

class CustomEstimator(override val uid: String) extends Estimator[CustomEstimatorModel]
  with DefaultParamsWritable
{
  override def fit(dataset: Dataset[_]): CustomEstimatorModel = {

    import dataset.sqlContext.implicits._

    val rddModel = dataset.rdd.pipe("src/main/python/train.py")
    val modelDF = rddModel.toDF()
    val text = modelDF.select(modelDF("value")).collect
    val textStr = text(0)(0).asInstanceOf[String]

    val model = new CustomEstimatorModel(uid, textStr)

    model
  }

  override def copy(extra: ParamMap): CustomEstimator = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(StructField("feature0",IntegerType,true),
      StructField("feature1",IntegerType,true),
      StructField("feature2",IntegerType,true),
      StructField("feature3",IntegerType,true),
      StructField("feature4",FloatType,true),
      StructField("feature5",IntegerType,true),
      StructField("feature6",FloatType,true),
      StructField("preds",FloatType,true)))
  }

  def this() = this(Identifiable.randomUID("CustomEstimator"))
}

object CustomEstimator extends DefaultParamsReadable[CustomEstimator] {
  override def load(path: String): CustomEstimator = super.load(path)
}

class CustomEstimatorModel(override val uid: String, val model: String) extends Model[CustomEstimatorModel]
 with MLWritable
{

  override def copy(extra: ParamMap): CustomEstimatorModel = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    import java.io._

    import dataset.sqlContext.implicits._

    val textStr = model
    val file = new File("python.model")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(textStr)
    bw.close()
    dataset.sparkSession.sparkContext.addFile("python.model", true)

    val rddRes = dataset.rdd.pipe(s"src/main/python/test.py")

    var res = rddRes.toDF()
    val split_col = split(res("value"), "[ ]+")

    res = res.withColumn("feature0", split_col.getItem(1))
    res = res.withColumn("feature1", split_col.getItem(2))
    res = res.withColumn("feature2", split_col.getItem(3))
    res = res.withColumn("feature3", split_col.getItem(4))
    res = res.withColumn("feature4", split_col.getItem(5))
    res = res.withColumn("feature5", split_col.getItem(6))
    res = res.withColumn("feature6", split_col.getItem(7))
    res = res.withColumn("preds", split_col.getItem(8))

    //delete row with headers
    res = res.filter(res("preds") =!= "preds")

    res = res.select(col("feature0"), col("feature1"), col("feature2"),
      col("feature3"), col("feature4"), col("feature5"), col("feature6"),
      col("preds"))

    res

  }

  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(StructField("feature0",IntegerType,true),
      StructField("feature1",IntegerType,true),
      StructField("feature2",IntegerType,true),
      StructField("feature3",IntegerType,true),
      StructField("feature4",FloatType,true),
      StructField("feature5",IntegerType,true),
      StructField("feature6",FloatType,true),
      StructField("preds",FloatType,true)))
  }

  override def write: MLWriter = new CustomEstimatorModelWriter(this)
}

object CustomEstimatorModel extends MLReadable[CustomEstimatorModel] {
  private[CustomEstimatorModel]
  class CustomEstimatorModelWriter(instance: CustomEstimatorModel) extends MLWriter {

    private case class Data(model: String)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.model)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class CustomEstimatorModelReader extends MLReader[CustomEstimatorModel] {

    private val className = classOf[CustomEstimatorModel].getName

    override def load(path: String): CustomEstimatorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
        .select("model")
        .head()
      val modelStr = data.getAs[String](0)
      val model = new CustomEstimatorModel(metadata.uid, modelStr)
      metadata.getAndSetParams(model)
      model
    }
  }

  override def read: MLReader[CustomEstimatorModel] = new CustomEstimatorModelReader

  override def load(path: String): CustomEstimatorModel = super.load(path)
}