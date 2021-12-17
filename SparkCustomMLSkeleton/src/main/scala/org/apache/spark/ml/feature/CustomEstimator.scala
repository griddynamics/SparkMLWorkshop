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

    // Pass your data into src/main/python/train.py and get the result into modelStr
    // val modelStr = ???

    val model = new CustomEstimatorModel(uid, modelStr)

    model
  }

  override def copy(extra: ParamMap): CustomEstimator = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    // Set StructType of your df
    // ???
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

    // Pass your data into src/main/python/test.py and get the result into res
    // val res = ???

    res

  }

  override def transformSchema(schema: StructType): StructType = {
    // Set StructType of your df
    // ???
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