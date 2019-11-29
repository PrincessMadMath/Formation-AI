using System;
using System.Linq;
using System.Text;

using Microsoft.ML.Data;

namespace Formation.AI
{
    public static class ClassificationModel
    {
        public class DataPoint
        {
            public bool Label { get; set; }

            // Size = number of features
            [VectorType(4)]
            public float[] Features { get; set; }
        }

        // TODO-3: Update features map here
        public static float[] GetFeatures(ActivityModel model)
        {
            // If you add feature don't forget to update the size of the Vector Size above
            return new float[]
            {
                model.CapturedDate.ToUnixTimeSeconds() -  model.LastCalendarActivityDate.ToUnixTimeSeconds(),
                model.CapturedDate.ToUnixTimeSeconds() - model.LastConversationActivityDate.ToUnixTimeSeconds(),
                model.CapturedDate.ToUnixTimeSeconds() - model.SiteCollectionLastActivityDate.ToUnixTimeSeconds(),
                model.CapturedDate.ToUnixTimeSeconds() - model.TeamsLastActivityDate.ToUnixTimeSeconds(),
            };
        }

        public class ActivityModel
        {
            public string TenantId { get; set; }

            public string GroupId { get; set; }

            public bool IsUnused { get; set; }

            public int GroupPrivacy { get; set; }

            public int OwnersCount { get; set; }

            public bool HasTeams { get; set; }

            public DateTimeOffset TeamsLastActivityDate { get; set; }

            public bool HasConversations { get; set; }

            public DateTimeOffset LastConversationActivityDate { get; set; }

            public bool HasEvents { get; set; }

            public DateTimeOffset LastCalendarActivityDate { get; set; }

            public bool HasSiteCollection { get; set; }

            public DateTimeOffset SiteCollectionLastActivityDate { get; set; }

            public DateTimeOffset CapturedDate { get; set; }
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

        public static DataPoint ConvertToDataPoint(ActivityModel model)
        {
            return new DataPoint()
            {
                Label = model.IsUnused,
                Features = GetFeatures(model)
            };
        }

    }
}