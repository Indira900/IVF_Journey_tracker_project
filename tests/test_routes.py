import pytest
from unittest.mock import patch, MagicMock
from flask import session


class TestRoutes:
    """Integration tests for Flask routes"""

    def test_home_page(self, client):
        """Test home page loads successfully"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'IVF Journey Tracker' in response.data

    def test_login_required_redirect(self, client):
        """Test that protected routes redirect to login"""
        response = client.get('/patient_dashboard')
        assert response.status_code == 302  # Redirect
        assert '/login' in response.headers['Location']

    def test_register_page(self, client):
        """Test registration page loads"""
        response = client.get('/register')
        assert response.status_code == 200
        assert b'Register' in response.data

    @patch('routes.User')
    def test_register_user_success(self, mock_user, client):
        """Test successful user registration"""
        mock_user.query.filter_by.return_value.first.return_value = None

        response = client.post('/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'first_name': 'Test',
            'last_name': 'User',
            'password': 'password123',
            'confirm_password': 'password123',
            'user_type': 'patient'
        }, follow_redirects=True)

        assert response.status_code == 200

    def test_login_page(self, client):
        """Test login page loads"""
        response = client.get('/login')
        assert response.status_code == 200
        assert b'Login' in response.data

    @patch('routes.User')
    @patch('routes.check_password_hash')
    def test_login_success(self, mock_check_password, mock_user, client):
        """Test successful login"""
        mock_user_obj = MagicMock()
        mock_user_obj.id = 1
        mock_user_obj.user_type = 'patient'
        mock_user.query.filter_by.return_value.first.return_value = mock_user_obj
        mock_check_password.return_value = True

        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'password123'
        }, follow_redirects=True)

        assert response.status_code == 200
        assert session.get('user_id') == 1

    def test_logout(self, client):
        """Test logout functionality"""
        with client:
            # First login
            with client.session_transaction() as sess:
                sess['user_id'] = 1

            response = client.get('/logout', follow_redirects=True)
            assert response.status_code == 200
            assert 'user_id' not in session

    @patch('routes.predict_and_store')
    def test_prediction_route(self, mock_predict, client):
        """Test IVF prediction route"""
        mock_predict.return_value = {
            'prediction': 1,
            'probability': 0.75,
            'prediction_id': 1
        }

        with client:
            with client.session_transaction() as sess:
                sess['user_id'] = 1

            response = client.get('/predict', follow_redirects=True)
            assert response.status_code == 200

    def test_ivf_predictor_page(self, client):
        """Test IVF predictor page loads"""
        response = client.get('/ivf_predictor')
        assert response.status_code == 200
        assert b'IVF Success Predictor' in response.data

    @patch('routes.Clinic')
    def test_find_clinic_page(self, mock_clinic, client):
        """Test find clinic page"""
        mock_clinic.query.all.return_value = []
        response = client.get('/find_clinic')
        assert response.status_code == 200
        assert b'Find IVF Clinics' in response.data

    def test_chatbot_page(self, client):
        """Test chatbot page loads"""
        response = client.get('/chatbot')
        assert response.status_code == 200
        assert b'IVF Assistant' in response.data

    def test_faq_page(self, client):
        """Test FAQ page loads"""
        response = client.get('/faq')
        assert response.status_code == 200
        assert b'FAQ' in response.data

    def test_wellness_page(self, client):
        """Test wellness page loads"""
        response = client.get('/wellness')
        assert response.status_code == 200
        assert b'Wellness Tracker' in response.data

    def test_partner_registration_page(self, client):
        """Test partner registration page loads"""
        response = client.get('/register/partner')
        assert response.status_code == 200
        assert b'Partner' in response.data

    def test_partner_dashboard_requires_login(self, client):
        """Test partner dashboard rejects unauthenticated access"""
        response = client.get('/partner/dashboard')
        assert response.status_code == 302
        assert '/login' in response.headers['Location']
