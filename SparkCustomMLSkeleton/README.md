# SparkCustomMLSkeletone

## Overview

This project contains skeleton code for building a pipeline and using custom classes based on Spark ML.

## Project Structure

    .
    └───SparkCustomMLPipeline                           # Project folder
        ├──project
        │  ├──build.properties                          # Build properties
        │  └──plugins.sbt                               # Build plugin: assembly for fat-jar              
        ├──src
        │  └──main
        │     ├──python
        │     │  ├──requirments.txt                     # Required python packages
        │     │  ├──test.py                             # python file for running model
        │     │  └──train.py                            # python file for fitting model
        │     └──scala
        │        ├──org.apache.spark.ml.feature         # This folder hierarchy is necessary for the correct operation of custom transformers and estimators
        │        │  ├──CustomEstimator.scala            # Custom estimator, which using python files for fitting and running model
        │        │  └──CustomTransformer.scala          # Custom transformer, which creating new field feature3, as a sum of feature1 and feature2
        │        ├──SimplepipelineExample.scala         # Example of standard spark ml-pipeline
        │        ├──Test.scala                          # Test pipeline
        │        └──Train.scala                         # Train pipeline
        ├──build.sbt                                    # Build properties   
        └──README.md                                    # Awesome README
