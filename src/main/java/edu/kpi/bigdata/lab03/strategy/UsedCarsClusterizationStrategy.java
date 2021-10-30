package edu.kpi.bigdata.lab03.strategy;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

public class UsedCarsClusterizationStrategy implements Serializable {

    private static final String APP_NAME = "Used Cars Clusterization Application";
    private static final int K = 30;
    private static final String FORMAT = "csv";
    private static final String INPUT_PATH = "/lab03/vehicles.csv";
    private static final String OUTPUT_PATH = "/lab03/results";
    private static final String PREDICTION_COLUMN = "prediction";
    private static final List<String> COLUMNS_TO_CLUSTER = Arrays.asList("price", "odometer", "year");
    private static final String FEATURES_COLUMN = "features";
    private static final String FEATURES_INPUT_COLUMN = "features_input";

    public void execute() {

        final SparkSession spark = createSparkSession();

        try {

            execute(spark);

        } catch (final Exception e) {

            e.printStackTrace();
        }

        spark.stop();
    }

    private SparkSession createSparkSession() {

        return SparkSession.builder()
                .appName(APP_NAME)
                .getOrCreate();
    }

    private void execute(final SparkSession spark) {

        final Dataset<Row> dataset = getDataset(spark);

        Optional.ofNullable(dataset)
                .map(this::trainModel)
                .map(models -> models.transform(dataset))
                .ifPresent(this::saveResult);
    }

    private Dataset<Row> selectResultColumns(final Dataset<Row> dataset) {

        return dataset.select(COLUMNS_TO_CLUSTER.get(0),
                Stream.concat(COLUMNS_TO_CLUSTER.stream().skip(1L), Stream.of(PREDICTION_COLUMN))
                        .toArray(String[]::new));
    }

    private Dataset<Row> getDataset(final SparkSession spark) {

        return Optional.ofNullable(spark)

                .map(this::getRawData)
                .map(this::filterData)
                .map(this::castColumns)
                .map(this::assembleColumns)
                .map(this::normalizeColumns)

                .orElseThrow(() -> new IllegalStateException("Dataset in null"));
    }

    private Dataset<Row> filterData(final Dataset<Row> dataset) {

        return dataset
                .filter(this::filterRow);
    }

    private boolean filterRow(final Row row) {

        return COLUMNS_TO_CLUSTER.stream()
                .allMatch(column -> filterFunction(row, column));
    }

    private boolean filterFunction(final Row row, final String fieldName) {

        try {

            return row.getAs(fieldName) != null
                    && Integer.parseInt(row.getAs(fieldName)) != 0;

        } catch (final Exception e) {

            return false;
        }
    }

    private Dataset<Row> getRawData(final SparkSession spark) {

        return spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(INPUT_PATH);
    }

    private Dataset<Row> castColumns(Dataset<Row> dataset) {

        for (String column : COLUMNS_TO_CLUSTER) {

            dataset = dataset.withColumn(column, new Column(column).cast("int"));
        }

        return dataset;
    }

    private Dataset<Row> normalizeColumns(Dataset<Row> dataset) {

        return new Normalizer()
                .setInputCol(FEATURES_INPUT_COLUMN)
                .setOutputCol(FEATURES_COLUMN)
                .transform(dataset.withColumnRenamed(FEATURES_COLUMN, FEATURES_INPUT_COLUMN))
                .drop(FEATURES_INPUT_COLUMN);
    }

    private Dataset<Row> assembleColumns(final Dataset<Row> dataset) {

        return new VectorAssembler()
                .setInputCols(COLUMNS_TO_CLUSTER.toArray(new String[0]))
                .setOutputCol(FEATURES_COLUMN)
                .setHandleInvalid("skip")
                .transform(dataset);
    }

    private KMeansModel trainModel(final Dataset<Row> dataset) {

        return new KMeans()
                .setK(K)
                .setSeed(1L)
                .fit(dataset);
    }

    private void saveResult(final Dataset<Row> dataset) {

        selectResultColumns(dataset)
                .write()
                .format(FORMAT)
                .mode(SaveMode.Overwrite)
                .save(OUTPUT_PATH);
    }
}
