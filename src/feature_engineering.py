from src.feature_engineering_pipeline import *


def transform_features(data):
    """
    Transform existing features for improved model performance.

    :param data: data with new features
    :return: data with transformed features

    Example Usage:
    --------------

    from feature_engineering_pipeline import Context, InteractionTerms, PolynomialFeatures, CustomOperation

    context = Context(data)

    # Interaction Terms
    interaction_terms = InteractionTerms(columns=['feature1', 'feature2'])
    interaction_terms.apply(context)

    # Polynomial Features
    polynomial_features = PolynomialFeatures(columns=['feature1', 'feature2'], degree=2)
    polynomial_features.apply(context)

    # ... (include other operations as needed)

    # Custom Operation
    def custom_function(ctx):
        # Define your custom transformation logic here
        ctx.data['new_feature'] = ctx.data['feature1'] * 10

    custom_operation = CustomOperation(custom_function)
    custom_operation.apply(context)

    # After applying all operations, return the transformed data
    return context.data

    """
    # TODO: Implement feature transformations
    pass


if __name__ == '__main__':
    import pandas as pd

    # Create a sample DataFrame
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8]
    })

    # Print the original data
    print("Original Data:")
    print(data)
    print()

    # Apply the feature transformations
    transformed_data = transform_features(data)

    # Print the transformed data
    print("Transformed Data:")
    print(transformed_data)

