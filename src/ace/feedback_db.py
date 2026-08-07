import sqlite3
import numpy as np
import pickle
import os
from datetime import datetime

class FeedbackDB:
    def __init__(self, db_path='data/ace_feedback.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                features BLOB,
                nir_confidence REAL,
                ace_threshold REAL,
                escalated INTEGER,
                nir_prediction TEXT,
                final_prediction TEXT,
                correct_class TEXT,
                mir_was_needed INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def record_prediction(self, features, nir_prediction, ace_threshold, escalated, final_prediction, polymer_class):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        feat_blob = pickle.dumps(features)
        ts = datetime.now().isoformat()
        
        c.execute('''
            INSERT INTO predictions (
                timestamp, features, nir_confidence, ace_threshold, 
                escalated, nir_prediction, final_prediction, correct_class, mir_was_needed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        ''', (
            ts, feat_blob, float(features[0]), float(ace_threshold),
            int(escalated), nir_prediction, final_prediction
        ))
        
        conn.commit()
        conn.close()

    def record_correction(self, record_id, correct_class):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT nir_prediction FROM predictions WHERE id = ?', (record_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return
            
        nir_pred = row[0]
        mir_needed = 1 if correct_class != nir_pred else 0
        
        c.execute('''
            UPDATE predictions 
            SET correct_class = ?, mir_was_needed = ? 
            WHERE id = ?
        ''', (correct_class, mir_needed, record_id))
        
        conn.commit()
        conn.close()

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT features, mir_was_needed FROM predictions WHERE mir_was_needed IS NOT NULL')
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            return np.array([]), np.array([])
            
        X = []
        y = []
        for row in rows:
            feat_blob, mir_needed = row
            feat = pickle.loads(feat_blob)
            X.append(feat)
            y.append(mir_needed)
            
        return np.array(X), np.array(y)

    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM predictions')
        total = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM predictions WHERE escalated = 1')
        escalated = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM predictions WHERE correct_class IS NOT NULL')
        corrected = c.fetchone()[0]
        
        conn.close()
        
        return {
            'total_predictions': total,
            'total_escalated': escalated,
            'escalation_rate': escalated / total if total > 0 else 0.0,
            'feedback_received': corrected
        }
