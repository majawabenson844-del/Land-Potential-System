elif page == "Predict":
    try:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.title("🔮 Predict Groundwater Potential")

        st.write("### Select values for the predictors:")

        # Country selection
        country = st.selectbox("Select Country", ["Select Country", "Zimbabwe"])

        if country == "Zimbabwe":
            # Province selection
            province = st.selectbox("Select Province", ["Select Province", "Midlands Province", "Masvingo Province"])
            
            # Define districts based on province
            districts = {
                "Midlands Province": ["Gweru", "Kwekwe", "Shurugwi"],
                "Masvingo Province": ["Masvingo", "Chiredzi", "Mwenezi"]
            }
            
            # Conditional district selection
            if province in districts:
                district = st.selectbox("Select District", ["Select District"] + districts[province])
            else:
                district = None  # If no valid province is selected
        else:
            district = None  # No district selection without country

        # Include input for user's location
        location = st.text_input("Your Location (Lat, Lon)", value="", placeholder="Automatically fetched location", key='location_input')

        user_inputs = {}

        # Assuming selected_features is defined earlier in the code
        for feature in selected_features:
            options = sorted(data[feature].dropna().unique().tolist())
            user_inputs[feature] = st.selectbox(f"🔸 {feature}", options)

        if st.button("✨ Predict Potential"):
            # Prediction logic here
            try:
                # Build the full feature vector
                full_input = default_values.copy()
                full_input.update(user_inputs)

                # Add location if available
                if location:
                    full_input['Location'] = location  # Ensure 'Location' is part of your model's features

                input_df = pd.DataFrame([full_input])[full_features]

                # Encode
                encoded = encoder.transform(input_df)
                encoded_df = pd.DataFrame(encoded, columns=full_features)

                # Select Boruta features
                selected_df = encoded_df[selected_features]

                # Scale
                scaled = scaler.transform(selected_df)

                # Predict
                pred = model.predict(scaled)[0]
                probs = model.predict_proba(scaled)[0]

                st.markdown("---")

                if pred == 1:
                    st.success("🌱 High Potential Area")
                else:
                    st.error("⚠️ Low Potential Area")

                col1, col2 = st.columns(2)
                col1.metric("High Potential Confidence", f"{probs[1] * 100:.2f}%")
                col2.metric("Low Potential Confidence", f"{probs[0] * 100:.2f}%")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error encountered: {e}")
