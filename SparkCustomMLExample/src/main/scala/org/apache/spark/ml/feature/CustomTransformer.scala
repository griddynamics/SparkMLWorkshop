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

  def hasColumn(df: Dataset[_], path: String) = Try(df(path)).isSuccess

  val exchangeRates = new FloatParam(this, "numFeatures", "number of features (> 0)", ParamValidators.gt(0))

  setDefault(exchangeRates -> (73.2).toFloat)

  def getExchangeRates: Float = $(exchangeRates)

  def setExchangeRates(value: Float): this.type = set(exchangeRates, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (hasColumn(dataset, "spendPerMonth")) {
     dataset.withColumnRenamed("id", "feature0")
        .withColumnRenamed("subscriptionType", "feature1")
        .withColumnRenamed("favoriteGenres", "feature2")
        .withColumnRenamed("promoCodes", "feature3")
        .withColumnRenamed("timePerMonth", "feature4")
        .withColumnRenamed("numberOfRatings", "feature5")
        .withColumnRenamed("spendPerMonth", "label")
        .withColumn("feature6", col("feature4") / 30)
        .withColumn("label", col("label") / getExchangeRates)
        .select(col("feature0"), col("feature1"), col("feature2"),
          col("feature3"), col("feature4"), col("feature5"),
          col("feature6"), col("label"))
    } else {
      dataset.withColumnRenamed("id", "feature0")
        .withColumnRenamed("subscriptionType", "feature1")
        .withColumnRenamed("favoriteGenres", "feature2")
        .withColumnRenamed("promoCodes", "feature3")
        .withColumnRenamed("timePerMonth", "feature4")
        .withColumnRenamed("numberOfRatings", "feature5")
        .withColumn("feature6", col("feature4") / 30)
        .select(col("feature0"), col("feature1"), col("feature2"),
          col("feature3"), col("feature4"), col("feature5"),
          col("feature6"))
    }
  }

  override def copy(extra: ParamMap): CustomTransformer = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = {
    if (schema.toString().contains("spendPerMonth")) {
      StructType(Seq(StructField("feature0", IntegerType, true),
        StructField("feature1", IntegerType, true),
        StructField("feature2", IntegerType, true),
        StructField("feature3", IntegerType, true),
        StructField("feature4", FloatType, true),
        StructField("feature5", IntegerType, true),
        StructField("feature6", FloatType, true),
        StructField("label", FloatType, true)))
    } else {
      StructType(Seq(StructField("feature0", IntegerType, true),
        StructField("feature1", IntegerType, true),
        StructField("feature2", IntegerType, true),
        StructField("feature3", IntegerType, true),
        StructField("feature4", FloatType, true),
        StructField("feature5", IntegerType, true),
        StructField("feature6", FloatType, true)))
    }
  }
}

object CustomTransformer extends DefaultParamsReadable[CustomTransformer] {
  override def load(path: String): CustomTransformer = super.load(path)
}