using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Formation.AI;

namespace Formation
{
    public class Program
    {
        // TODO-1: Change path to the data file
        private static string dataPath = @"C:\tmp\data-28-11-2019\output.txt";
        private static string modelSavePath = @"C:\Temp\inactive-model";

        public static async Task Main()
        {
            var trainer = GetTrainer();

            Train(trainer);
            DoExam();
        }

        public static BinaryClassificationTrainer GetTrainer()
        {
            // TODO-2: Choose your algorithm type
            return new BinaryClassificationTrainer(BinaryClassificationTrainer.AlgorithmType.FastTree);
        }

        public static void Train(ITrainer<ClassificationModel.DataPoint, bool> trainer)
        {
            var loader = new ModelLoader<ClassificationModel.DataPoint>(ClassificationModel.ConvertToDataPoint);

            var (trainingSet, testSet) = loader.LoadData(dataPath);

            trainer.Train(trainingSet);

            trainer.ComputeMetrics(testSet);

            trainer.Save(modelSavePath);
        }

        public static void DoExam()
        {
            var trainer = GetTrainer();

            trainer.Load(modelSavePath);

            var score = 0;

            foreach (var data in ExamData)
            {
                var isInactive = trainer.Predict(ClassificationModel.ConvertToDataPoint(data.Data));
                if (isInactive == data.IsUnused)
                {
                    ++score;
                }
            }

            Console.WriteLine($"Score: {score}/{ExamData.Count}");

            trainer.Save(modelSavePath);
        }

        private static List<(ClassificationModel.ActivityModel Data, bool IsUnused)> ExamData = new List<(ClassificationModel.ActivityModel, bool)>
        {
            (new ClassificationModel.ActivityModel()
            {
                TenantId = Guid.NewGuid().ToString(),
                GroupId = Guid.NewGuid().ToString(),
                GroupPrivacy = 1,
                OwnersCount = 2,
                HasEvents = true,
                LastCalendarActivityDate = DateTimeOffset.Parse("2019-11-28T00:00:00.000Z"),
                HasConversations = true,
                LastConversationActivityDate = DateTimeOffset.Parse("2019-11-28T00:00:00.000Z"),
                HasTeams = true,
                TeamsLastActivityDate = DateTimeOffset.Parse("2019-11-28T00:00:00.000Z"),
                HasSiteCollection = true,
                SiteCollectionLastActivityDate =  DateTimeOffset.Parse("2019-11-28T00:00:00.000Z"),
                CapturedDate =  DateTimeOffset.Parse("2020-11-29T00:00:00.000Z"),
            }, true),
            (new ClassificationModel.ActivityModel()
            {
                TenantId = Guid.NewGuid().ToString(),
                GroupId = Guid.NewGuid().ToString(),
                GroupPrivacy = 2,
                OwnersCount = 3,
                HasEvents = false,
                LastCalendarActivityDate = DateTimeOffset.MinValue,
                HasConversations = false,
                LastConversationActivityDate = DateTimeOffset.MinValue,
                HasTeams = false,
                TeamsLastActivityDate =  DateTimeOffset.MinValue,
                HasSiteCollection = false,
                SiteCollectionLastActivityDate =   DateTimeOffset.MinValue,
                CapturedDate =  DateTimeOffset.Parse("2019-11-29T00:00:00.000Z"),
            }, true),
            (new ClassificationModel.ActivityModel()
            {
                TenantId = Guid.NewGuid().ToString(),
                GroupId = Guid.NewGuid().ToString(),
                GroupPrivacy = 1,
                OwnersCount = 1,
                HasEvents = false,
                LastCalendarActivityDate = DateTimeOffset.MinValue,
                HasConversations = false,
                LastConversationActivityDate = DateTimeOffset.MinValue,
                HasTeams = false,
                TeamsLastActivityDate =  DateTimeOffset.MinValue,
                HasSiteCollection = true,
                SiteCollectionLastActivityDate =  DateTimeOffset.Parse("2019-11-20T00:00:00.000Z"),
                CapturedDate =  DateTimeOffset.Parse("2019-11-29T00:00:00.000Z"),
            }, false),
            (new ClassificationModel.ActivityModel()
            {
                TenantId = Guid.NewGuid().ToString(),
                GroupId = Guid.NewGuid().ToString(),
                GroupPrivacy = 1,
                OwnersCount = 4,
                HasEvents = true,
                LastCalendarActivityDate = DateTimeOffset.Parse("2010-11-28T00:00:00.000Z"),
                HasConversations = true,
                LastConversationActivityDate = DateTimeOffset.Parse("2011-11-28T00:00:00.000Z"),
                HasTeams = true,
                TeamsLastActivityDate = DateTimeOffset.Parse("2019-11-28T00:00:00.000Z"),
                HasSiteCollection = true,
                SiteCollectionLastActivityDate =  DateTimeOffset.Parse("2010-11-28T00:00:00.000Z"),
                CapturedDate =  DateTimeOffset.Parse("2019-11-29T00:00:00.000Z"),
            }, false),
            (new ClassificationModel.ActivityModel()
            {
                TenantId = Guid.NewGuid().ToString(),
                GroupId = Guid.NewGuid().ToString(),
                GroupPrivacy = 1,
                OwnersCount = 4,
                HasEvents = true,
                LastCalendarActivityDate = DateTimeOffset.Parse("2019-11-28T00:00:00.000Z"),
                HasConversations = true,
                LastConversationActivityDate = DateTimeOffset.Parse("2011-11-28T00:00:00.000Z"),
                HasTeams = true,
                TeamsLastActivityDate = DateTimeOffset.Parse("2010-11-28T00:00:00.000Z"),
                HasSiteCollection = true,
                SiteCollectionLastActivityDate =  DateTimeOffset.Parse("2010-11-28T00:00:00.000Z"),
                CapturedDate =  DateTimeOffset.Parse("2019-11-29T00:00:00.000Z"),
            }, false),
        };
    }
}