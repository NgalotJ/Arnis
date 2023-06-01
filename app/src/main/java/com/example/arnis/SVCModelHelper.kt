package com.example.arnis

import android.content.Context
import android.util.Log
import libsvm.svm
import libsvm.svm_model
import libsvm.svm_node
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream

class SVCModelHelper(context: Context) {
    companion object {
        const val TAG = "SVCModelHelper"
        private const val MODEL_FILENAME = "svc_model.pkl"
    }

    private var model: svm_model? = null

    init {
        try {
            // Load the SVM model from assets folder
            val modelFile = File(context.filesDir, MODEL_FILENAME)
            val inputStream: InputStream = context.assets.open(MODEL_FILENAME)
            val outputStream: OutputStream = FileOutputStream(modelFile)
            inputStream.copyTo(outputStream)
            outputStream.close()
            inputStream.close()

            // Load the SVM model from the file
            model = svm.svm_load_model(modelFile.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading SVM model: ${e.message}")
        }
    }

    fun predict(features: DoubleArray): Int {
        val nodes = Array(features.size) { svm_node() }
        for (i in features.indices) {
            nodes[i] = svm_node().apply {
                index = i
                value = features[i]
            }
        }

        val probabilities = DoubleArray(model?.nr_class ?: 0)
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