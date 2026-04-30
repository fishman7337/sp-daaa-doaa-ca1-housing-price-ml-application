"""WTForms definitions for authentication and prediction."""

from __future__ import annotations

from typing import Any

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, MultipleFileField
from wtforms import (
    BooleanField,
    FloatField,
    IntegerField,
    PasswordField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms import SelectField
from wtforms.validators import Email, InputRequired, Length, Optional, NumberRange


class LoginForm(FlaskForm):
    """Login form."""

    username = StringField("Username", validators=[InputRequired(), Length(max=80)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=6)])
    remember = BooleanField("Remember me")
    submit = SubmitField("Login")


class SignupForm(FlaskForm):
    """Signup form."""

    username = StringField("Username", validators=[InputRequired(), Length(max=80)])
    email = StringField("Email", validators=[InputRequired(), Email(), Length(max=120)])
    password = PasswordField("Password", validators=[InputRequired(), Length(min=6)])
    terms = BooleanField(
        "I agree to the Terms of Use and Privacy Policy",
        validators=[InputRequired(message="Please accept the terms to continue.")],
    )
    submit = SubmitField("Create account")


class PredictionForm(FlaskForm):
    """Prediction form covering structured, text, and image inputs."""

    city = SelectField(
        "City",
        validators=[InputRequired(message="Select a valid city")],
        choices=[],
        coerce=str,
        render_kw={"data-placeholder": "Search city"},
    )
    state = SelectField(
        "State",
        validators=[InputRequired(message="Select a valid state")],
        choices=[],
        coerce=str,
        render_kw={"data-placeholder": "Search state"},
    )
    status = SelectField(
        "Status",
        validators=[InputRequired(message="Select a valid status")],
        choices=[("for_sale", "For Sale"), ("sold", "Sold"), ("ready_to_build", "Ready to Build")],
        coerce=str,
    )
    bedrooms = FloatField(
        "Bedrooms",
        validators=[
            InputRequired(message="Enter a positive number"),
            NumberRange(min=0.0001, message="Enter a positive number"),
        ],
    )
    bathrooms = FloatField(
        "Bathrooms",
        validators=[
            InputRequired(message="Enter a positive number"),
            NumberRange(min=0.0001, message="Enter a positive number"),
        ],
    )
    living_area = FloatField(
        "Living Area (sqft)",
        validators=[
            InputRequired(message="Enter a positive number"),
            NumberRange(min=0.0001, message="Enter a positive number"),
        ],
    )
    lot_size = FloatField(
        "Lot Size (sqft)",
        validators=[
            InputRequired(message="Enter a positive number"),
            NumberRange(min=0.0001, message="Enter a positive number"),
        ],
    )
    description = TextAreaField(
        "Description", validators=[Optional(), Length(max=5000)]
    )
    image = MultipleFileField(
        "Property Image",
        validators=[Optional(), FileAllowed(["jpg", "jpeg", "png", "webp"])],
        render_kw={"accept": "image/*"},
    )
    submit = SubmitField("Predict")

    def validate(self, extra_validators: Any | None = None) -> bool:
        """Require all structured fields; description/images remain optional."""

        return super().validate(extra_validators=extra_validators)


__all__ = ["LoginForm", "SignupForm", "PredictionForm"]
