using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Formation.AI;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace Formation
{
    // Legacy exemple using BinaryClassifier
    public class BinaryClassificationTrainer
    {
        public class DataPoint
        {
            public bool Label { get; set; }

            // Size = number of features
            [VectorType(1)]
            public float[] Features { get; set; }
        }
        
        public class BinaryPrediction: DataPoint
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedLabel { get; set; }

            [ColumnName("Probability")]
            public float Probability { get; set; }

            [ColumnName("Score")]
            public float Score { get; set; }
        }

        public static DataPoint ConvertToDataPoint(ActivityData data)
        {
            return new DataPoint()
            {
                Label = data.IsUnused,
                Features = GetFeatures(data)
            };
        }
        
        public static float[] GetFeatures(ActivityData data)
        {
            // If you add feature don't forget to update the size of the Vector Size above
            return new float[]
            {
                data.HasTeams ? 1 : 0,
            };
        }
        
        
        private readonly AlgorithmType _algorithmType;

        private MLContext _mlContext;
        private ITransformer _trainedModel;
        private DataViewSchema _schema;
        private PredictionEngine<DataPoint, BinaryPrediction> _predictionEngine;

        public BinaryClassificationTrainer(AlgorithmType algorithmType)
        {
            this._algorithmType = algorithmType;
            this._mlContext = new MLContext(0);
        }

        public enum AlgorithmType
        {
            Svm,
            AveragePerceptron,
            FastForest,
            FastTree,
        }

        public void Train(IEnumerable<DataPoint> trainingSet)
        {
            this._mlContext = new MLContext(0);

            var trainingDataView = this._mlContext.Data.LoadFromEnumerable(trainingSet);
            this._schema = trainingDataView.Schema;

            IEstimator<ITransformer> trainingPipeline;
            switch (this._algorithmType)
            {
                case AlgorithmType.Svm:
                    trainingPipeline = this.BuildSvm();
                    this._trainedModel = trainingPipeline.Fit(trainingDataView);
                    break;
                case AlgorithmType.AveragePerceptron:
                    trainingPipeline = this.BuildAveragePerceptron();
                    this._trainedModel = trainingPipeline.Fit(trainingDataView);
                    break;
                case AlgorithmType.FastForest:
                    trainingPipeline = this.BuildFastForest();
                    this._trainedModel = trainingPipeline.Fit(trainingDataView);
                    break;
                case AlgorithmType.FastTree:
                    trainingPipeline = this.BuildFastTree();
                    this._trainedModel = trainingPipeline.Fit(trainingDataView);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            this._predictionEngine = this._mlContext.Model.CreatePredictionEngine<DataPoint, BinaryPrediction>(this._trainedModel);
        }

        public void ComputeMetrics(IEnumerable<DataPoint> testSet)
        {
            var testDataView = this._mlContext.Data.LoadFromEnumerable(testSet);

            var computed = this._trainedModel.Transform(testDataView);

            var metrics = this._mlContext.BinaryClassification.EvaluateNonCalibrated (computed);

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            // Which is the proportion of correct predictions in the test set.
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");

            // Indicates how confident the model is correctly classifying the positive and negative classes
            // Closer to 1 the better
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");

            // Measure of balance between precision and recall
            // Closer to 1 the better
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        public Task Save(string path)
        {
            this._mlContext.Model.Save(this._trainedModel, this._schema, path);

            return Task.CompletedTask;
        }

        public Task Load(string path)
        {
            var loadedModel = this._mlContext.Model.Load(path, out var modelInputSchema);

            this._predictionEngine = this._mlContext.Model.CreatePredictionEngine<DataPoint, BinaryPrediction>(loadedModel);

            return Task.CompletedTask;
        }

        public bool Predict(DataPoint dataPoint)
        {
            return this._predictionEngine.Predict(dataPoint).PredictedLabel;
        }

        public IEstimator<ITransformer> BuildSvm()
        {
            var pipeline = this._mlContext.BinaryClassification.Trainers.LinearSvm();

            return pipeline;
        }

        public IEstimator<ITransformer> BuildAveragePerceptron()
        {
            var pipeline = this._mlContext.BinaryClassification.Trainers.AveragedPerceptron();

            return pipeline;
        }

        public IEstimator<ITransformer> BuildFastForest()
        {
            var pipeline = this._mlContext.BinaryClassification.Trainers.FastForest();

            return pipeline;
        }

        public IEstimator<ITransformer> BuildFastTree()
        {
            var pipeline = this._mlContext.BinaryClassification.Trainers.FastTree();

            return pipeline;
        }
    }
}