using System;

namespace Formation.AI
{
    public class ActivityData
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
}