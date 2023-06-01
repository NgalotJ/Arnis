package com.example.arnis

import android.content.Context
import android.util.Log
import libsvm.svm
import libsvm.svm_model
import libsvm.svm_node
import org.apache.commons.io.IOUtils
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation
import org.apache.commons.math3.stat.inference.TTest
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.sqrt


class SVCModelHelper(context: Context) {
    companion object {
        const val TAG = "SVCModelHelper"
        private const val MODEL_FILENAME = "svc_model.pkl"
    }

    private lateinit var model: svm_model

    init {
        try {
            // Load the SVM model from assets folder
            val modelFile = File(context.filesDir, MODEL_FILENAME)
            val inputStream: InputStream = context.assets.open(MODEL_FILENAME)
            val outputStream: OutputStream = FileOutputStream(modelFile)
            IOUtils.copy(inputStream, outputStream)
            outputStream.close()
            inputStream.close()

            // Load the SVM model from the file
            model = svm.svm_load_model(modelFile.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading SVM model: ${e.message}")
        }
    }

    fun predict(features: DoubleArray): Int {
        val scaledFeatures = scaleFeatures(features)
        val selectedFeatures = selectFeatures(scaledFeatures)
        return svmPredict(selectedFeatures)
    }

    private fun scaleFeatures(features: DoubleArray): DoubleArray {
        val correlationMatrix = calculateCorrelationMatrix(features)
        val scalingFactors = calculateScalingFactors(correlationMatrix)

        val scaledFeatures = DoubleArray(features.size)
        for (i in features.indices) {
            scaledFeatures[i] = features[i] / scalingFactors[i]
        }

        return scaledFeatures
    }

    private fun calculateCorrelationMatrix(features: DoubleArray): Array<DoubleArray> {
        val correlation = PearsonsCorrelation()
        val correlationMatrix = Array(features.size) { DoubleArray(features.size) }

        for (i in features.indices) {
            for (j in features.indices) {
                val feature1 = doubleArrayOf(features[i])
                val feature2 = doubleArrayOf(features[j])
                correlationMatrix[i][j] = correlation.correlation(feature1, feature2)
            }
        }

        return correlationMatrix
    }

    private fun calculateScalingFactors(correlationMatrix: Array<DoubleArray>): DoubleArray {
        val scalingFactors = DoubleArray(correlationMatrix.size)
        for (i in correlationMatrix.indices) {
            var varianceSum = 0.0
            for (j in correlationMatrix.indices) {
                if (i != j) {
                    varianceSum += sqrt(1.0 - correlationMatrix[i][j] * correlationMatrix[i][j])
                }
            }
            scalingFactors[i] = sqrt(1.0 / varianceSum)
        }
        return scalingFactors
    }

    private fun selectFeatures(features: DoubleArray): DoubleArray {
        val k = 55 // Number of top features to select

        val scores = computeScores(features) // Compute feature scores
        val topFeatureIndices = selectTopKFeatures(scores, k) // Select top k features

        val selectedFeatures = DoubleArray(k)
        for (i in topFeatureIndices.indices) {
            selectedFeatures[i] = features[topFeatureIndices[i]]
        }

        return selectedFeatures
    }

    private fun computeScores(features: DoubleArray): DoubleArray {
        val labels = arrayOf(
            "Right Temple Strike",
            "Stomach Thrust",
            "Left Knee Strike"
        )

        val tScores = DoubleArray(features.size)

        for (i in features.indices) {
            val feature = features[i]
            val label = labels[i % labels.size] // Assign labels in a cyclic manner
            val tTest = TTest()
            tScores[i] = tTest.t(feature, getLabelValues(label)) // Compute t-score for the feature
        }

        return tScores
    }

    private fun getLabelValues(label: String): DoubleArray {
        // Assign numerical values to each label for t-test comparison
        return when (label) {
            "Right Temple Strike" -> doubleArrayOf(0.0)
            "Stomach Thrust" -> doubleArrayOf(1.0)
            "Left Knee Strike" -> doubleArrayOf(2.0)
            else -> throw IllegalArgumentException("Unknown label: $label")
        }
    }

    private fun selectTopKFeatures(scores: DoubleArray, k: Int): IntArray {
        val indexedScores = scores.mapIndexed { index, score -> index to score }
        val sortedIndices = indexedScores.sortedByDescending { it.second }
            .subList(0, k)
            .map { it.first }
            .toIntArray()

        return sortedIndices
    }

    private fun svmPredict(features: DoubleArray): Int {
        val nodes = Array(features.size) { svm_node() }
        for (i in features.indices) {
            nodes[i] = svm_node().apply {
                index = i
                value = features[i]
            }
        }

        val probabilities = DoubleArray(model.nr_class)
        svm.svm_predict_probability(model, nodes, probabilities)

        var maxIndex = 0
        var maxProbability = probabilities[0]
        for (i in 1 until probabilities.size) {
            if (probabilities[i] > maxProbability) {
                maxIndex = i
                maxProbability = probabilities[i]
            }
        }

        return maxIndex
    }
}