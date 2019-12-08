using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Bomberjam.Bot;
using Bomberjam.Bot.AI;
using Formation.AI;

namespace Formation
{
    public class Program
    {
        // TODO-1: Change path to the data file
        private static string dataPath = @"F:\tmp\output.txt";
        private static string modelSavePath = @"F:\tmp\inactive-model.zip";
        
        enum ProgramRole {
            TrainModel,
            EvaluateFeatures,
            DoExam,
        }


        public static async Task Main()
        {
            // TODO-2: Choose if you want to train or do the exam
            var role = ProgramRole.TrainModel;
            
            var smartBot = new RawSmartBot(MulticlassAlgorithmType.LightGbm, int.MaxValue);
            
            switch (role)
            {
                case ProgramRole.TrainModel:
                    await TestModel(smartBot);
                    break;
                case ProgramRole.EvaluateFeatures:
                    EvaluateFeatures(smartBot);
                    break;
                case ProgramRole.DoExam:
                    DoExam(smartBot);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static async Task TestModel<T>(ISmartBot<T> smartBot) where T : LabeledDataPoint
        {
            smartBot.Train(dataPath, true);
            await smartBot.Save(modelSavePath);
        }
        
        // https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/explain-machine-learning-model-permutation-feature-importance-ml-net
        // Currently only support impact LightGbm algo
        public static void EvaluateFeatures<T>(ISmartBot<T> smartBot) where T : LabeledDataPoint
        {
            smartBot.EvaluateFeatures(dataPath);
        }
        
        public static void DoExam<T>(ISmartBot<T> smartBot) where T : LabeledDataPoint
        {

            smartBot.Load(modelSavePath);

            var score = 0;

            foreach (var data in ExamData)
            {
                var isInactive = smartBot.Predict(smartBot.ExtractDataPoint(data.Data));
                if (isInactive == data.IsUnused)
                {
                    ++score;
                }
            }

            Console.WriteLine($"Score: {score}/{ExamData.Count}");
        }

        private static List<(ActivityData Data, bool IsUnused)> ExamData = new List<(ActivityData, bool)>
        {
            (new ActivityData()
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
            (new ActivityData()
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
            (new ActivityData()
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
            (new ActivityData()
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
            (new ActivityData()
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