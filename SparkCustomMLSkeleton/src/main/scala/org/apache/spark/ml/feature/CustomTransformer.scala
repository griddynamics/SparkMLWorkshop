package org.apache.spark.ml.feature

import org.apache.spark.annotation.Since
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{FloatParam, IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.util.Try

class CustomTransformer(override val uid: String) extends Transformer
  with DefaultParamsWritable
{

  def this() = this(Identifiable.randomUID("org.apache.spark.ml.feature.CustomTransformer"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    // Set transformations of your df here
    // ???
  }

  override def copy(extra: ParamMap): CustomTransformer = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = {
    // Set StructType of your df
    // ???
  }
}

object CustomTransformer extends DefaultParamsReadable[CustomTransformer] {
  override def load(path: String): CustomTransformer = super.load(path)
}