// Luma graph schema: constraints and indexes.

// ---- Identity ----
CREATE CONSTRAINT entity_id      IF NOT EXISTS FOR (n:Entity)     REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT event_id       IF NOT EXISTS FOR (n:Event)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT session_id     IF NOT EXISTS FOR (n:Session)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT image_id       IF NOT EXISTS FOR (n:Image)      REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT audioclip_id   IF NOT EXISTS FOR (n:AudioClip)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT textblock_id   IF NOT EXISTS FOR (n:TextBlock)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT detection_id   IF NOT EXISTS FOR (n:Detection)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT utterance_id   IF NOT EXISTS FOR (n:Utterance)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT entitytype_key IF NOT EXISTS FOR (n:EntityType) REQUIRE n.key IS UNIQUE;

// ---- Lookup / time-range indexes (used by queries and the pruning job) ----
CREATE INDEX entity_name       IF NOT EXISTS FOR (n:Entity)     ON (n.name);
CREATE INDEX event_at          IF NOT EXISTS FOR (n:Event)      ON (n.at);
CREATE INDEX image_captured    IF NOT EXISTS FOR (n:Image)      ON (n.captured_at);
CREATE INDEX detection_created IF NOT EXISTS FOR (n:Detection)  ON (n.created_at);
CREATE INDEX utterance_created IF NOT EXISTS FOR (n:Utterance)  ON (n.created_at);
CREATE INDEX textblock_created IF NOT EXISTS FOR (n:TextBlock)  ON (n.created_at);

// ---- Full-text ----
CREATE FULLTEXT INDEX entity_text    IF NOT EXISTS FOR (n:Entity)    ON EACH [n.name, n.description];
CREATE FULLTEXT INDEX event_text     IF NOT EXISTS FOR (n:Event)     ON EACH [n.title, n.text, n.context];
CREATE FULLTEXT INDEX textblock_text IF NOT EXISTS FOR (n:TextBlock) ON EACH [n.text];
CREATE FULLTEXT INDEX utterance_text IF NOT EXISTS FOR (n:Utterance) ON EACH [n.text];
