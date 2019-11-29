using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Formation.AI
{
    public interface ITrainer<T, TP>
    {
        void Train(IEnumerable<T> trainingSet);

        void ComputeMetrics(IEnumerable<T> testSet);

        Task Save(string path);

        Task Load(string path);

        TP Predict(T dataPoint);
    }
}