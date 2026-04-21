"""
CyberShield v4 — SQLAlchemy Database Models
4 tables: users, complaints, velocity_db, pending_review
"""
import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


# ─────────────────────────────────────────────
# TABLE 1 — users
# ─────────────────────────────────────────────
class User(db.Model, UserMixin):
    __tablename__ = "users"

    id         = db.Column(db.String(128), primary_key=True)   # Google sub ID
    google_id  = db.Column(db.String(128), unique=True)
    email      = db.Column(db.String(256), unique=True, nullable=False)
    name       = db.Column(db.String(256))
    picture    = db.Column(db.String(512))                     # Google avatar URL
    is_admin   = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    last_login = db.Column(db.DateTime)

    complaints = db.relationship("Complaint",     backref="user", lazy=True)
    pending    = db.relationship("PendingReview", backref="user", lazy=True)

    def __repr__(self):
        return f"<User {self.email}>"


# ─────────────────────────────────────────────
# TABLE 2 — complaints
# ─────────────────────────────────────────────
class Complaint(db.Model):
    __tablename__ = "complaints"

    id                  = db.Column(db.Integer, primary_key=True, autoincrement=True)
    complaint_id        = db.Column(db.String(16), unique=True, nullable=False)
    user_id             = db.Column(db.String(128), db.ForeignKey("users.id"), nullable=True)
    timestamp           = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    message             = db.Column(db.Text, nullable=False)
    risk_level          = db.Column(db.String(16))
    fraud_probability   = db.Column(db.Float)
    fraud_category      = db.Column(db.String(128))
    ncrp_code           = db.Column(db.String(16))

    # JSON-encoded lists
    phones_found        = db.Column(db.Text)   # json.dumps([...])
    urls_found          = db.Column(db.Text)
    amounts_found       = db.Column(db.Text)
    otps_found          = db.Column(db.Text)
    keywords            = db.Column(db.Text)
    velocity_alerts     = db.Column(db.Text)   # json.dumps([...])

    complainant_name    = db.Column(db.String(256))
    complainant_contact = db.Column(db.String(32))
    sender              = db.Column(db.String(128))
    pdf_path            = db.Column(db.String(512))

    pending_review      = db.relationship("PendingReview",
                                          foreign_keys="PendingReview.complaint_id",
                                          primaryjoin="Complaint.complaint_id == PendingReview.complaint_id",
                                          backref="complaint_obj", lazy=True)

    def __repr__(self):
        return f"<Complaint {self.complaint_id} {self.risk_level}>"


# ─────────────────────────────────────────────
# TABLE 3 — velocity_db
# ─────────────────────────────────────────────
class VelocityEntry(db.Model):
    __tablename__ = "velocity_db"

    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    indicator_type  = db.Column(db.String(16), nullable=False)    # "phone" | "url"
    indicator_value = db.Column(db.String(512), nullable=False)   # normalised value
    count           = db.Column(db.Integer, default=1)
    complaint_ids   = db.Column(db.Text)                          # JSON array
    first_seen      = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    last_seen       = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("indicator_type", "indicator_value", name="uq_type_value"),
    )

    def __repr__(self):
        return f"<VelocityEntry {self.indicator_type}:{self.indicator_value} x{self.count}>"


# ─────────────────────────────────────────────
# TABLE 4 — pending_review
# ─────────────────────────────────────────────
class PendingReview(db.Model):
    __tablename__ = "pending_review"

    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    complaint_id     = db.Column(db.String(16), db.ForeignKey("complaints.complaint_id"))
    user_id          = db.Column(db.String(128), db.ForeignKey("users.id"), nullable=True)
    message          = db.Column(db.Text, nullable=False)
    fraud_probability= db.Column(db.Float)
    admin_label      = db.Column(db.String(16), default="")   # "fraud" | "legit" | ""
    labeled_by       = db.Column(db.String(256))
    labeled_at       = db.Column(db.DateTime)
    created_at       = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<PendingReview {self.complaint_id} label={self.admin_label!r}>"
