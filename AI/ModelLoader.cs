using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

using Newtonsoft.Json;

namespace Formation.AI
{
    public class ModelLoader<T>
    {
        private const int SampleSize = int.MaxValue;
        private const double TestRatio = 0.2;

        public ModelLoader(GenerateDataPointDelegate generateDataPoint)
        {
            this.GenerateDataPoint = generateDataPoint;
        }

        public delegate T GenerateDataPointDelegate(ClassificationModel.ActivityModel step);

        public GenerateDataPointDelegate GenerateDataPoint { get; }

        public (IEnumerable<T> trainingSet, IEnumerable<T> testSet) LoadData(string path)
        {
            var models = JsonConvert.DeserializeObject<List<ClassificationModel.ActivityModel>>(File.ReadAllText(path))
                .Select(
                    x =>
                    {
                        x.CapturedDate = DateTimeOffset.Parse("2019-11-28T00:00:00.000Z");
                        return x;
                    });

            return this.SplitData(models);
        }

        private (IEnumerable<T> trainingSet, IEnumerable<T> testSet) SplitData(
            IEnumerable<ClassificationModel.ActivityModel> models)
        {
            var enumerable = models.Take(SampleSize).ToList();
            var dataCount = enumerable.Count();
            var testCount = (int)Math.Ceiling(dataCount * TestRatio);
            var trainingCount = dataCount - testCount;

            var trainingSet = enumerable.Take(trainingCount).Select(x => this.GenerateDataPoint(x));
            var testSet = enumerable.TakeLast(testCount).Select(x => this.GenerateDataPoint(x));

            return (trainingSet, testSet);
        }
    }
}