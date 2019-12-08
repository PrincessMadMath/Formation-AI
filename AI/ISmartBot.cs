using System.Threading.Tasks;
using Bomberjam.Bot;

namespace Formation.AI
{
    public interface ISmartBot<T> where T: LabeledDataPoint
    {
        T ExtractDataPoint(ActivityData data, bool? label = null);
        
        // Create model + output metrics
        void Train(string gameLogsPath, bool calculateMetrics = false);

        Task Save(string path);

        Task Load(string path);

        bool Predict(T dataPoint);
        
        void EvaluateFeatures(string gameLogsPath);
    }
}