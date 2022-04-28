from pydantic import BaseModel, Field


class CensusData(BaseModel):
    """ Datamodel for census income prediction """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str 
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 53,
                "workclass": "Private",
                "fnlgt": 234721,
                "education": "11th",
                "education_num": 7,
                "marital_status": "Married-civ-spouse",
                "occupation": "Handlers-cleaners",
                "relationship": "Husband",
                "race": "Black",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }
