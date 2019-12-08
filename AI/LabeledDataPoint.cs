using Microsoft.ML.Data;

namespace Bomberjam.Bot
{
    public abstract class LabeledDataPoint
    {
        public bool Label { get; set; }
    }
}