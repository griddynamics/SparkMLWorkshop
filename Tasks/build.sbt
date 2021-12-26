lazy val scala = "2.12.10"
lazy val sparkVersion = "3.1.2"

lazy val commonSettings = Seq(
  version := "1.0.0",
  scalaVersion := scala,
  unmanagedBase := baseDirectory.value / "lib",
  resolvers ++= Seq(
    Resolver.mavenLocal,
    Resolver.sonatypeRepo("releases"),
    Resolver.DefaultMavenRepository
  ),
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion
  )
)

lazy val root = project
  .in(file("."))
  .settings(commonSettings: _*)
  .aggregate(
    task1,
    task2,
    task3
  )

lazy val task1 = (project in file("task1"))
  .settings(commonSettings: _*)
  .enablePlugins(AssemblyPlugin)
  .settings(
    name := "task1",
    version := "1.0.0",
    assembly / test := {},
    assembly / assemblyMergeStrategy := {
      case PathList(ps@_*) if ps.last.endsWith("src/main/resources") => MergeStrategy.discard
      case PathList("META-INF", _*) => MergeStrategy.discard
      case _ => MergeStrategy.first
    },
    assembly / assemblyOption := (assembly / assemblyOption).value.copy(includeScala = false, includeDependency = false)
  )

lazy val task2 = (project in file("task2"))
  .settings(commonSettings: _*)
  .enablePlugins(AssemblyPlugin)
  .settings(
    name := "task2",
    version := "1.0.0",
    assembly / test := {},
    assembly / assemblyMergeStrategy := {
      case PathList(ps@_*) if ps.last.endsWith("src/main/resources") => MergeStrategy.discard
      case PathList("META-INF", _*) => MergeStrategy.discard
      case _ => MergeStrategy.first
    },
    assembly / assemblyOption := (assembly / assemblyOption).value.copy(includeScala = false, includeDependency = false)
  )

lazy val task3 = (project in file("task3"))
  .settings(commonSettings: _*)
  .enablePlugins(AssemblyPlugin)
  .settings(
    name := "task3",
    version := "1.0.0",
    assembly / test := {},
    assembly / assemblyMergeStrategy := {
      case PathList(ps@_*) if ps.last.endsWith("src/main/resources") => MergeStrategy.discard
      case PathList("META-INF", _*) => MergeStrategy.discard
      case _ => MergeStrategy.first
    },
    assembly / assemblyOption := (assembly / assemblyOption).value.copy(includeScala = false, includeDependency = false)
  )
