import streamlit as st
import pickle
from preprocess import preprocess_input

# Streamlit app configuration
st.set_page_config(page_title="AI Profit Maximizer", layout="wide", page_icon="📊")

# Load the trained model
@st.cache_resource
def load_model():
    with open('../trained_model/pipeline.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Title and description
st.title("AI Profit Maximizer 📊")
st.markdown("### Predict property values and make informed decisions.")

# Input section
st.markdown("---")
st.subheader("Provide Property Details")

# Input fields
crime_rate = st.number_input("Crime Rate (e.g., 2.65):", value=2.65, step=0.01)
renovation_level = st.selectbox("Renovation Level:", [
    "Minor", "Extensive", "Complete", "Essential", "Advanced", "Basic", "Standard", "Limited", "Premium", "Partial"
])
num_rooms = st.selectbox("Number of Rooms:", ["2", "3", "4", "6", "8"], format_func=lambda x: f"{x} rooms")
property_type = st.selectbox("Property Type:", ["1", "2", "3", "4", "8"], format_func=lambda x: {
    "1": "Single Family", "2": "Two Family", "3": "Three Family", "4": "Four Family", "8": "Condo"
}[x])
amenities_rating = st.selectbox("Amenities Rating:", [
    "Outstanding", "Superb", "Luxurious", "Satisfactory", "Mediocre", "Exceptional", "Marginal", "Unimpressive", "Commonplace", "Below Average"
])
carpet_area = st.number_input("Carpet Area (in sqft):", value=760.0, step=1.0)
property_tax_rate = st.number_input("Property Tax Rate (%):", value=1.03, step=0.01)
locality = st.selectbox("Locality:", [
    "Greenwich", "East Hampton", "Ridgefield", "Old Lyme", "Naugatuck", "Killingly", "Glastonbury", "Bridgeport",
    "Danbury", "Hamden", "Branford", "Norwalk", "East Granby", "Windsor", "Oxford", "Chester", "Thompson", "Eastford",
    "Newington", "Harwinton", "Canterbury", "West Haven", "Waterbury", "Bristol", "Bloomfield", "Plainfield",
    "Wallingford", "Farmington", "Berlin", "Ellington", "Stratford", "Woodstock", "Fairfield", "New Britain",
    "Waterford", "Redding", "Stonington", "Derby", "Sprague", "East Haddam", "Cromwell", "New Milford",
    "Southington", "Wethersfield", "South Windsor", "New London", "Windham", "Trumbull", "Newtown", "Winchester",
    "Durham", "Wilton", "Middletown", "Putnam", "Windsor Locks", "Preston", "Suffield", "Roxbury", "Granby",
    "Marlborough", "Tolland", "Simsbury", "Avon", "Essex", "Darien", "Madison", "Sterling", "Haddam", "Somers",
    "Cheshire", "Torrington", "Rocky Hill", "Westport", "Ashford", "Ledyard", "Stafford", "Voluntown", "Bethlehem",
    "Seymour", "Norwich", "Lyme", "Guilford", "North Branford", "Bolton", "Bethel", "Burlington", "New Fairfield",
    "Watertown", "Woodbridge", "Litchfield", "Sharon", "Kent", "Weston", "Brooklyn", "Sherman", "East Windsor",
    "New Canaan", "Southbury", "Colebrook", "Woodbury", "New Hartford", "North Haven", "Barkhamsted", "Willington",
    "Monroe", "Deep River", "Clinton", "Mansfield", "Ansonia", "New Haven", "Warren", "Chaplin", "Washington",
    "Old Saybrook", "Bethany", "Canton", "Prospect", "Coventry", "North Stonington", "Hampton", "Lebanon",
    "Salisbury", "Morris", "Norfolk", "Union", "Goshen", "North Canaan", "Franklin", "Canaan", "Pomfret",
    "Scotland", "Hartland", "Bozrah", "Milford", "East Hartford", "Westbrook", "Salem", "Brookfield", "Wolcott",
    "Manchester", "East Haven", "West Hartford", "Plainville", "Vernon", "Orange", "Portland", "Groton",
    "Shelton", "Columbia", "Hartford", "Thomaston", "Meriden", "Easton", "Enfield", "Griswold", "Cornwall",
    "East Lyme", "Lisbon", "Hebron", "Montville", "Colchester", "Killingworth", "Bridgewater", "Andover",
    "Plymouth", "Middlebury", "Beacon Falls", "Middlefield", "***Unknown***", "Stamford"
])
residential = st.selectbox("Residential Type:", [
    "Condominium", "Detached House", "Triplex", "Duplex", "Fourplex"
])
estimated_value = st.number_input("Estimated Value (in $):", value=711270.0, step=1000.0)

# Prediction button
if st.button("Predict Value"):
    try:
        # Prepare input data
        input_data = {
            "crime_rate": crime_rate,
            "renovation_level": renovation_level,
            "num_rooms": num_rooms,
            "Property": property_type,
            "amenities_rating": amenities_rating,
            "carpet_area": carpet_area,
            "property_tax_rate": property_tax_rate,
            "Locality": locality,
            "Residential": residential,
            "Estimated Value": estimated_value
        }

        # Preprocess input data
        preprocessed_data = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(preprocessed_data)[0]
        st.success(f"The predicted property value is: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
