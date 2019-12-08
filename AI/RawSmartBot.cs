using System;
using System.Collections.Generic;
using Formation.AI;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Bomberjam.Bot.AI
{
    // Bot using raw features
    public class RawSmartBot: BaseSmartBot<RawSmartBot.RawActivityData>
    {
        private const int featuresSize = 1;
        
        // Datapoint
        public class RawActivityData: LabeledDataPoint
        {
            // Size = number of features
            [VectorType(featuresSize)]
            public float[] Features { get; set; }
        }
        

        public RawSmartBot(MulticlassAlgorithmType algorithmType, int sampleSize = 100): base(algorithmType, sampleSize)
        {
        }

        // TODO-3: Select the features of the model
        protected override RawActivityData ExtractFeatures(ActivityData data)
        {
            var rdm = new Random();
            
            // TODO: If you add or remove feature don't forget to update the `featuresSize` variable
            var features = new List<float>
            {
//                data.HasTeams ? 1 : 0,
                rdm.Next()
            };
            
            if (features.Count != featuresSize)
            {
                Console.WriteLine($"Feature count does not match, expected {featuresSize}, received {features.Count}");
                throw new ArgumentOutOfRangeException();
            }
            
            return new RawActivityData()
            {

                Features = features.ToArray()
            };
        }

        // Because the dataPoint has the format expected by ML.Net (one column Features and one Label)
        // their is no need to transform the data.
        protected override IEnumerable<IEstimator<ITransformer>> GetFeaturePipeline()
        {
            return Array.Empty<IEstimator<ITransformer>>();
        }
    }
}